"""Builds the ComfyUI workflow "BP Inpainting - pictures to model" (comfyui/workflows/) from ComfyUI's
object_info (the description of every node a running ComfyUI knows).

    Pictures (+ own pictures / video) -> Sort (data sheet) -> Base model -> parts from easy to hard
    -> Combine -> Hair -> Compare (colour and clay), with a picture preview after every node and a 3D
    preview after the base, the combined model and the hair.

Usage (ComfyUI must be running, with the bp_inpainting nodes linked in):
    python3 comfyui/build_workflow.py                       # asks http://127.0.0.1:8188
    python3 comfyui/build_workflow.py --url http://127.0.0.1:8189 --install
"""
import argparse
import json
import os
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}
nodes, links, groups = {}, [], []


class Ref(tuple):
    pass


def ref(nid, slot=0):
    return Ref((nid, slot))


def node(cls, pos, title=None, size=(320, 120), mode=0, **kw):
    nid = len(nodes) + 1
    nodes[nid] = dict(cls=cls, pos=pos, title=title, size=size, mode=mode, kw=kw)
    return nid


def group(title, x, y, w, h, color):
    groups.append({"id": len(groups) + 1, "title": title, "bounding": [x, y, w, h], "color": color,
                   "font_size": 26, "flags": {}})


def spec(info, cls):
    if cls not in info:
        return []
    d = info[cls]["input"]
    out = []
    for name, s in list(d.get("required", {}).items()) + list(d.get("optional", {}).items()):
        t, opts = s[0], (s[1] if len(s) > 1 else {})
        is_widget = ((isinstance(t, list) or t in WIDGET_TYPES or opts.get("widgetType") in WIDGET_TYPES)
                     and not opts.get("forceInput"))
        out.append((name, "COMBO" if isinstance(t, list) else t, opts, is_widget, t))
    return out


def default_of(t, opts):
    if "default" in opts:
        return opts["default"]
    if isinstance(t, list):
        return t[0] if t else ""
    return {"INT": 0, "FLOAT": 0.0, "STRING": "", "BOOLEAN": False}.get(t)


def build(info):
    """The workflow for the ComfyUI page, and the same as an API prompt (for running it from a script)."""
    ui_nodes, link_id, outs, api = [], 0, {}, {}
    for nid, n in nodes.items():
        u_in, wv, a_in = [], [], {}
        for name, t, opts, is_widget, raw in spec(info, n["cls"]):
            v = n["kw"].get(name)
            if isinstance(v, Ref):
                link_id += 1
                src, slot = v
                links.append([link_id, src, slot, nid, len(u_in), info[nodes[src]["cls"]]["output"][slot]])
                outs.setdefault(v, []).append(link_id)
                a_in[name] = [str(src), slot]
                entry = {"name": name, "type": t, "link": link_id}
                if is_widget:
                    entry["widget"] = {"name": name}
                    wv.append(default_of(raw, opts))
                u_in.append(entry)
            elif is_widget:
                wv.append(default_of(raw, opts) if v is None else v)
                a_in[name] = wv[-1]
            else:
                u_in.append({"name": name, "type": t, "link": None})
            if is_widget and opts.get("control_after_generate"):
                wv.append("fixed")
            if is_widget and (opts.get("image_upload") or opts.get("video_upload")):
                wv.append("image" if opts.get("image_upload") else "video")
        if n["cls"] == "MarkdownNote":
            wv = [n["kw"]["text"]]
        n["u_in"], n["wv"] = u_in, wv
        if n["cls"] != "MarkdownNote" and n["mode"] == 0:
            api[str(nid)] = {"class_type": n["cls"], "inputs": a_in, "_meta": {"title": n["title"] or n["cls"]}}
    for nid, n in nodes.items():
        info_cls = info.get(n["cls"], {})
        types, names = info_cls.get("output", []), info_cls.get("output_name", [])
        u_out = [{"name": names[i] if i < len(names) else t, "type": t, "links": outs.get((nid, i), []), "slot_index": i}
                 for i, t in enumerate(types)]
        un = {"id": nid, "type": n["cls"], "pos": list(n["pos"]), "size": list(n["size"]), "flags": {}, "order": nid,
              "mode": n["mode"], "inputs": n["u_in"], "outputs": u_out,
              "properties": {"Node name for S&R": n["cls"]}, "widgets_values": n["wv"]}
        if n["title"]:
            un["title"] = n["title"]
        ui_nodes.append(un)
    # a switched-off (bypassed) node passes its first input on: link around it in the API prompt
    off = {str(nid): n for nid, n in nodes.items() if n["mode"] == 4}
    for entry in api.values():
        for name, v in entry["inputs"].items():
            while isinstance(v, list) and v[0] in off:
                v = [str(off[v[0]]["kw"]["pictures"][0]), off[v[0]]["kw"]["pictures"][1]]
            entry["inputs"][name] = v
    ui = {"last_node_id": len(nodes), "last_link_id": link_id, "nodes": ui_nodes, "links": links, "groups": groups,
          "config": {}, "extra": {"ds": {"scale": 0.4, "offset": [80, 160]}}, "version": 0.4}
    return ui, api


NOTE = """## BP Inpainting: pictures -> 3D model

**1 Pictures**: the character's folder (relative to the project). Own pictures or a video? Switch
**Add picture** / **Add video** on (Ctrl+B) and upload in them.

**2 Data sheet**: the vision AI sorts every view: shape only (grey), colour only (other version),
which part each close-up shows. Check the sheet before the GPU spends time.

**3 Base model**: the whole figure (TRELLIS.2), with renders and a clay render (shape alone).

**4 Parts**, from easy to hard: each part zoomed in from its one sharpest picture. *auto* = colours
only (shape kept); the head also regenerates its shape, from one picture in total. The face is a
lottery, so it is made with 4 seeds and the local AI (Qwen3.6-27B) picks the best. Back and butt are
left out (no good colour picture of them yet).

**5 Final**: Combine (the head wins where parts overlap) -> Hair strands (Blender Geometry Nodes)
-> Compare: base vs final, in colour and as clay.

ComfyUI only re-runs what changed. Every result also stays in `outputs/pipeline/<name>/<node>/`,
so a restart computes nothing twice. The GPU runs one node at a time."""


def layout():
    pictures = node("BP_Pictures", (0, 0), "1 Pictures", name="morrigan", folder="pictures/local model")
    added = node("BP_AddPicture", (0, 200), "Add picture (switch on to use)", mode=4, pictures=ref(pictures))
    video = node("BP_AddVideo", (0, 560), "Add video (switch on to use)", mode=4, pictures=ref(added), frames=12)
    node("PreviewImage", (360, 0), "all pictures", size=(420, 380), images=ref(pictures, 1))
    node("MarkdownNote", (-560, 0), "How to use", size=(520, 620), text=NOTE)
    group("1 Pictures", -20, -60, 820, 860, "#3f789e")

    sort = node("BP_Sort", (860, 0), "2 Sort pictures (data sheet)", pictures=ref(video))
    node("PreviewImage", (860, 180), "data sheet", size=(620, 560), images=ref(sort, 1))
    node("PreviewAny", (1500, 0), "lists", size=(420, 740), source=ref(sort, 2))
    group("2 Data sheet", 840, -60, 1100, 860, "#88a")

    base = node("BP_BaseModel", (1980, 0), "3 Base model (TRELLIS.2)", size=(320, 220), sorted=ref(sort))
    node("PreviewImage", (1980, 260), "base: pictures, renders, clay", size=(560, 480), images=ref(base, 1))
    node("Preview3D", (2560, 0), "base in 3D", size=(420, 420), model_file=ref(base, 2))
    group("3 Base model", 1960, -60, 1040, 860, "#a1309b")

    parts = {}
    for k, part in enumerate(["lower_legs", "thighs", "feet", "belly", "bust"]):
        x = 3040 + k * 520
        parts[part] = node("BP_RefinePart", (x, 0), f"4.{k + 1} {part}", size=(340, 240), base=ref(base), sorted=ref(sort),
                           part=part)
        node("PreviewImage", (x, 280), f"{part}: pictures | before | after", size=(500, 460), images=ref(parts[part], 1))
    group("4 Parts, from easy to hard", 3020, -60, 5 * 520 + 20, 860, "#b58b2a")
    # the face is a lottery (r8 clean, the same inputs again broken): 4 seeds, the local AI picks the best
    heads = []
    for k, seed in enumerate((42, 7, 123, 2024)):
        x = 3040 + k * 520
        heads.append(node("BP_RefinePart", (x, 900), f"4.6 head, seed {seed}", size=(340, 240), base=ref(base),
                          sorted=ref(sort), part="head", seed=seed))
        node("PreviewImage", (x, 1180), f"head seed {seed}", size=(500, 460), images=ref(heads[-1], 1))
    parts["head"] = node("BP_PickBest", (3040 + 4 * 520, 900), "4.7 Pick best head (local AI)", size=(320, 200),
                         **{f"candidate_{k + 1}": ref(h) for k, h in enumerate(heads)})
    node("PreviewImage", (3040 + 4 * 520, 1140), "the local AI's choice", size=(620, 500), images=ref(parts["head"], 1))
    group("4b The face last: 4 seeds, the local AI picks", 3020, 840, 4 * 520 + 700, 860, "#b06634")

    x = 3040 + 6 * 520 + 60
    order = ["head", "bust", "belly", "thighs", "lower_legs", "feet"]  # the first wins where two overlap
    combine = node("BP_Combine", (x, 0), "5 Combine", size=(320, 260), base=ref(base),
                   **{f"part_{i + 1}": ref(parts[p]) for i, p in enumerate(order)})
    node("PreviewImage", (x, 300), "base | combined", size=(520, 440), images=ref(combine, 1))
    node("Preview3D", (x + 540, 0), "final in 3D", size=(440, 440), model_file=ref(combine, 2))
    hair = node("BP_Hair", (x, 780), "6 Hair strands", size=(320, 200), model=ref(combine), sorted=ref(sort),
                head=ref(parts["head"]))
    node("PreviewImage", (x + 340, 780), "hair", size=(520, 420), images=ref(hair, 0))
    node("Preview3D", (x + 880, 780), "final with hair in 3D", size=(440, 420), model_file=ref(hair, 1))
    compare = node("BP_Compare", (x + 1000, 0), "99 Compare (colour)", size=(300, 120), model_1=ref(base),
                   model_2=ref(combine), clay=False)
    node("PreviewImage", (x + 1000, 160), "base vs final", size=(560, 560), images=ref(compare, 0))
    clay = node("BP_Compare", (x + 1580, 0), "99 Compare (clay: the shape alone)", size=(300, 120), model_1=ref(base),
                model_2=ref(combine), clay=True)
    node("PreviewImage", (x + 1580, 160), "shape: base vs final", size=(560, 560), images=ref(clay, 0))
    group("5 Final: combine, hair, compare", x - 20, -60, 2180, 1300, "#3f8f5f")


def main():
    parser = argparse.ArgumentParser(description="Build the BP Inpainting ComfyUI workflow")
    parser.add_argument("--url", default="http://127.0.0.1:8188", help="a running ComfyUI")
    parser.add_argument("--install", action="store_true", help="also copy it into ComfyUI's saved workflows")
    parser.add_argument("--comfyui", default=os.path.expanduser("~/ai-models/tools/ComfyUI"))
    args = parser.parse_args()
    info = json.load(urllib.request.urlopen(f"{args.url}/object_info"))
    missing = [c for c in ("BP_Pictures", "BP_Sort", "BP_BaseModel", "BP_RefinePart") if c not in info]
    if missing:
        raise SystemExit(f"ComfyUI doesn't know {missing}: is comfyui/bp_inpainting linked into custom_nodes?")
    layout()
    workflow, api = build(info)
    out = os.path.join(HERE, "workflows", "BP Inpainting - pictures to model.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(workflow, open(out, "w"), indent=1)
    json.dump(api, open(out.replace(".json", ".api.json"), "w"), indent=1)
    print(f"Wrote {out} ({len(nodes)} nodes, {len(links)} links)")
    if args.install:
        target = os.path.join(args.comfyui, "user", "default", "workflows", os.path.basename(out))
        json.dump(workflow, open(target, "w"), indent=1)
        print(f"Installed {target}")


if __name__ == "__main__":
    main()
