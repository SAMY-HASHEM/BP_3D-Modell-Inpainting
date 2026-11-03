"""Many pictures of one character -> one detailed model, as a chain of nodes (like Geometry Nodes).

Every node has its own folder, settings and outputs, and can be re-run alone. The nodes and their settings
are all in NODES and ROUNDS below, so the whole pipeline can be read in one place.

    01_sort          the vision AI sorts the pictures into lists (sort_pictures.py)
    02_base          the whole figure: shape from one version's views, colours from the colour views
    04_refine_<part> one part zoomed in and repainted from all its pictures (refine_region.py)
    05_combine       the refined parts put into the base model (combine_parts.py)
    06_hair          real hair strands grown on the hair shell (hair_strands.py, Blender Geometry Nodes)
    99_compare       side by side: reference | before | after, the same camera (compare.jpg per node,
                     overview.jpg of the latest round of every node, history.jpg of all rounds)

Rounds: ROUNDS holds what changed in each round and why. A node's results go to <node>/r<round>/.
Later nodes take the newest round of the nodes before them.

Usage (TRELLIS.2 environment, everything on this computer's GPU):
    python scripts/pipeline/run_pipeline.py "pictures/local model" --name morrigan --round 1
    python scripts/pipeline/run_pipeline.py "pictures/local model" --name morrigan --round 2 --only 04_refine_head
    python scripts/pipeline/run_pipeline.py "pictures/local model" --name morrigan --compare-only
"""
import _paths  # noqa: F401  (the scripts of the other folders)

import argparse
import glob
import json
import os
import subprocess
import sys
import time

from PIL import Image, ImageDraw

PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPTS = os.path.join(PROJECT, "scripts")
PYTHON = sys.executable
BLENDER = os.environ.get("BLENDER", "blender")

# ------------------------------------------------------------------ the nodes and their settings

NODES = {
    "01_sort": {"model": "Qwen/Qwen3-VL-8B-Instruct"},
    # night tests: stochastic best and 2x faster; 1024 (1536 makes holes in hair); 12 steps = 25; 4M faces keep detail
    "02_base": {"mode": "stochastic", "resolution": "1024", "steps": 12, "faces": 4_000_000, "seed": 42},
    # every part, with the lessons of rounds 1-9 (see LESSONS and docs/pipeline-nodes.md):
    #   colours only (keep_shape): the base shape from the grey views is already good (legs r6 got lumpy)
    #   one sharp colour picture (best_colour 1): mixing pictures smeared the face (r4 -> r8)
    #   margin as a share of the model (0.1): big parts still get zoomed (chest r5 was the whole model)
    #   texture guidance 3: the colours follow the picture (r3 -> r4: pink lips instead of black)
    "04_refine": {"method": "repaint", "margin": 0.6, "steps": 25, "mode": "stochastic", "seed": 42, "grow": 1,
                  "texture_guidance": 3.0, "faces": 4_000_000, "margin_of_model": 0.1, "keep_shape": True,
                  "best_colour": 1,
                  # the order in 05_combine (first wins where two regions overlap)
                  "parts": ["head", "bust", "belly", "butt", "back", "thighs", "lower_legs", "feet", "legs"]},
    # the head is the only part whose shape is regenerated too (zoomed: hair locks, ears, lips), margin 0.6
    "04_refine_head": {"keep_shape": False, "margin_of_model": None},
}

LESSONS = [  # what the rounds taught, as rules the code follows now (round -> rule)
    "r2: all colour close-ups in the base mix the colours up -> the base gets the whole colour view only",
    "r5: a back view in a front part turns the region around -> only pictures from the main view's side",
    "r5: a margin of 0.6 of a big part is the whole model -> the margin as a share of the model (0.1)",
    "r6: regenerating the shape of big parts from close-ups makes them lumpy -> colours only (keep shape)",
    "r3/r4: black make-up -> texture guidance 3; r8: one sharp picture beats six mixed ones -> best_colour 1",
    "r9: the largest picture can be a cut-off close-up from below -> the largest COMPLETE picture",
    "r9: back / butt need their main view from behind -> part side (sort_pictures.PART_SIDE)",
    "r9: back.jpg has another skin colour -> it must not give colours (not solved: the AI said same colours)",
    "best of r1-r9 combined: colour bands where two parts meet -> next: colour once over the whole model",
    "THE USER'S PRINCIPLE: everything is made without colour first (like Roman marble statues: shape only, "
    "judged in clay renders), the colours only at the very end, from the pictures, in ONE pass over the "
    "finished model, so both sides and all parts match (no colour seams, symmetric)",
]

NODES["06_hair"] = {"count": 30000, "length": 0.10, "points": 16, "hue_tolerance": 0.07}

ROUNDS = {  # round: {node: changed settings}, "why": the reason
    1: {"why": "first run with the settings from the night tests and the head test"},
    2: {"why": "r1: shape very good, colours too dark (leggings dark red instead of pink, hair dark teal): the colour "
               "stage also gets the colour close-ups of the same character, which show the colours much larger",
        "02_base": {"colour_closeups": True}},
    3: {"why": "r2: hair greener and leggings pink, but the orange gloves lost and the suit patchy (9 colour pictures "
               "in 12 steps: each only 1-2 steps). The base stays r1; the parts get their colours from their own "
               "close-ups in the refine nodes",
        "02_base": {"colour_closeups": False}, "use": {"02_base": 1}},
    4: {"why": "r3 head: hair much better (green, fringe, locks), but the face got black eye make-up and black lips "
               "(the pictures: pink lips): the colours follow the pictures more strongly (texture guidance 3), and "
               "the region keeps 4M faces (r3 saved 1M where the base had 1.4M)",
        "04_refine_head": {"texture_guidance": 3.0, "faces": 4_000_000}},
    5: {"why": "r4 head: pink lips and eye shadow like the pictures, brighter hair: the same settings for every part",
        **{f"04_refine_{part}": {"texture_guidance": 3.0, "faces": 4_000_000} for part in ("chest", "back", "legs", "feet")},
        "use": {"04_refine_head": 4}},
    6: {"why": "r5 chest failed: the back picture (seen from behind) in the chest list turned the body around, and "
               "margin 0.6 of the chest = almost the whole model (no zoom). Now: only pictures from the main view's "
               "side, the margin as a share of the model (0.1) for every part but the head, and the parts from easy "
               "to hard (legs, feet, chest, back; the face last)",
        **{f"04_refine_{part}": {"texture_guidance": 3.0, "faces": 4_000_000, "margin_of_model": 0.1}
           for part in ("chest", "back", "legs", "feet")},
        "use": {"04_refine_head": 4}},
}

ROUNDS[7] = {"why": "r6 legs: the colours right (pink with bats) but the shape worse (lumpy, a hole at the knee); "
                    "the base shape from the 3 grey views was already good. Now two kinds of refine: colour only "
                    "(keep shape) for the clothes and body, shape + colours only for the head",
             **{f"04_refine_{part}": {"texture_guidance": 3.0, "faces": 4_000_000, "margin_of_model": 0.1,
                                      "keep_shape": True} for part in ("chest", "back", "legs", "feet")},
             "use": {"04_refine_head": 4}}

ROUNDS[8] = {"why": "face test: r3/r4 mixed 6 head pictures from different angles (stochastic: each colour picture "
                    "~8 of 25 steps), which smears eyes and lips; the ninja head worked with ONE picture. Now the "
                    "head with one sharp picture only (morrigan-01-001), like the ninja",
             "04_refine_head": {"texture_guidance": 3.0, "faces": 4_000_000,
                                "only": ["deyvson-william-morrigan-01-camera-money-shot-001"]}}

ROUNDS[9] = {"why": "the user: bust, belly and butt looked bad. They were inside the big chest / legs boxes (2x and "
                    "1.5x zoom) and the butt on the seam between back and legs. Now finer parts (bust, belly, butt, "
                    "thighs, lower legs) with ~4-5x zoom; grey / colour measured (002 was wrongly 'grey'); "
                    "colours only, the shape stays",
             **{f"04_refine_{part}": {"texture_guidance": 3.0, "faces": 4_000_000, "margin_of_model": 0.1,
                                      "keep_shape": True, "best_colour": 1}
                for part in ("bust", "belly", "butt", "back", "thighs", "lower_legs", "feet")},
             "use": {"04_refine_head": 8}}
ROUNDS[9]["why"] += "; r8 head with ONE picture: clean lips and eyes, so every part takes only its sharpest colour picture"

ROUNDS[10] = {"why": "the best of rounds 1-9 combined (the user: the best full model): head r8, bust / belly / "
                     "thighs / feet r9, the lower legs from legs r7; back r9 (brown skin from back.jpg), butt r9 (almost "
                     "white) and lower legs r9 (a cut-off close-up from below) are left out",
              "use": {"04_refine_head": 8, "04_refine_bust": 9, "04_refine_belly": 9, "04_refine_thighs": 9,
                      "04_refine_feet": 9, "04_refine_legs": 7, "04_refine_back": 0, "04_refine_butt": 0,
                      "04_refine_lower_legs": 0}}

ROUNDS_OF = "morrigan"  # ROUNDS is Morrigan's history; another character starts from NODES (the lessons) alone

# from easy to hard (the user: legs and clothes first, the face last)
RUN_ORDER = ["lower_legs", "thighs", "feet", "belly", "back", "butt", "bust", "head"]


def settings(node, round_):
    """A node's settings in a round: NODES, changed by every round up to this one."""
    key = "04_refine" if node.startswith("04_refine") else node
    result = dict(NODES.get(key, {}))
    if node != key:
        result.update(NODES.get(node, {}))  # a part's own settings (e.g. the head)
    for r in sorted(ROUNDS):
        if r <= round_:
            result.update(ROUNDS[r].get(node, {}))
    return result


# ------------------------------------------------------------------ helpers

class Pipeline:
    def __init__(self, pictures, name, round_):
        self.pictures, self.round = pictures, round_
        self.root = os.path.join(PROJECT, "outputs", "pipeline", name)

    def folder(self, node, round_=None):
        path = os.path.join(self.root, node, f"r{round_ or self.round}")
        os.makedirs(path, exist_ok=True)
        return path

    def newest(self, node, pattern):
        """The newest round's file of an earlier node (later nodes build on it), or the round a round chose
        with "use" in ROUNDS."""
        use = {}
        for r in sorted(ROUNDS):
            if r <= self.round:
                use.update(ROUNDS[r].get("use", {}))
        if node in use and use[node] == 0:
            return None  # this round leaves the node out (its results got worse)
        if node in use:
            found = sorted(glob.glob(os.path.join(self.root, node, f"r{use[node]}", pattern)))
            if found:
                return found[0]
        for r in sorted((int(d[1:]) for d in os.listdir(os.path.join(self.root, node)) if d.startswith("r")), reverse=True) \
                if os.path.isdir(os.path.join(self.root, node)) else []:
            found = sorted(glob.glob(os.path.join(self.root, node, f"r{r}", pattern)))
            if found:
                return found[0]
        return None

    def run(self, node, command, outputs, info=None):
        """Runs one node (unless its outputs are there), with its log and node.json."""
        folder = self.folder(node)
        if all(os.path.exists(o) for o in outputs):
            print(f"[{node}] done before (r{self.round})", flush=True)
            return True
        print(f"[{node}] r{self.round}: {' '.join(os.path.relpath(c, PROJECT) if c.startswith(PROJECT) else c for c in command)}",
              flush=True)
        start = time.time()
        with open(os.path.join(folder, "log.txt"), "w") as log:
            code = subprocess.call(command, stdout=log, stderr=subprocess.STDOUT, cwd=PROJECT,
                                   env={**os.environ, "PYTHONUNBUFFERED": "1"})
        ok = code == 0 and all(os.path.exists(o) for o in outputs)
        json.dump({"node": node, "round": self.round, "why": ROUNDS.get(self.round, {}).get("why"),
                   "settings": settings(node, self.round), "command": command, "outputs": outputs,
                   "seconds": round(time.time() - start), "ok": ok, **(info or {})},
                  open(os.path.join(folder, "node.json"), "w"), indent=1)
        print(f"[{node}] {'ok' if ok else 'FAILED, see ' + os.path.join(folder, 'log.txt')} "
              f"({time.time() - start:.0f} s)", flush=True)
        return ok


def render(model, out, focus=None, size=384, clay=False):
    """Preview renders with Blender on the CPU (render_preview.py); returns the picture paths. With a focus box:
    front, 45° left and right, and from behind (r9: back and butt were never seen from behind)."""
    extra = ["--focus", *[f"{v:.5f}" for v in focus]] if focus is not None else []
    extra += ["--clay"] if clay else []  # shape only (the shape part of the pipeline is judged without colours)
    names = ["front", "left", "right", "back"] if focus is not None else ["0", "1", "2", "3"]
    paths = [f"{out}_{n}.png" for n in names]
    if not all(os.path.exists(p) for p in paths):
        subprocess.call([BLENDER, "-b", "--python", os.path.join(SCRIPTS, "tools", "render_preview.py"), "--",
                         os.path.abspath(model), os.path.abspath(out), "--size", str(size), *extra],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return [p for p in paths if os.path.exists(p)]


def sheet(rows, out, cell=320):
    """A comparison picture: rows of (label, [pictures]); every picture fitted into one cell."""
    width = max(len(pics) for _, pics in rows) * cell
    picture = Image.new("RGB", (width, len(rows) * (cell + 26)), (40, 40, 44))
    draw = ImageDraw.Draw(picture)
    for r, (label, pics) in enumerate(rows):
        y = r * (cell + 26)
        draw.text((6, y + 6), label, fill=(255, 255, 255))
        for c, p in enumerate(pics):
            image = Image.open(p).convert("RGB") if isinstance(p, str) else p
            image.thumbnail((cell, cell))
            picture.paste(image, (c * cell + (cell - image.width) // 2, y + 26 + (cell - image.height) // 2))
    picture.save(out, quality=90)
    return out


# ------------------------------------------------------------------ the nodes

def node_01_sort(p):
    folder = p.folder("01_sort")
    out = os.path.join(folder, "sorted.json")
    ok = p.run("01_sort", [PYTHON, os.path.join(SCRIPTS, "pipeline", "sort_pictures.py"), p.pictures, "--out", folder,
                           "--model", settings("01_sort", p.round)["model"]], [out])
    if ok:
        sheet([("01_sort: every view with the vision AI's tags (contact.jpg)", [os.path.join(folder, "contact.jpg")])],
              os.path.join(folder, "compare.jpg"), cell=1400)
    return ok


def sorted_result(p):
    return json.load(open(p.newest("01_sort", "sorted.json")))


def view_path(p, view):
    return os.path.join(os.path.dirname(p.newest("01_sort", "sorted.json")), view["view"])


def node_02_base(p):
    s = settings("02_base", p.round)
    result = sorted_result(p)
    views = {v["name"]: v for v in result["views"]}
    lists = result["lists"]["base"]
    path = lambda names: [view_path(p, views[n]) for n in names]
    if s.get("colour_closeups"):  # every colour view of the same character and colours, close-ups too
        parts = [result["lists"][k] for k in ("head", "hair", "chest", "back", "legs", "feet") if k in result["lists"]]
        more = [n for part in parts for n in part["both"] + part["colour_only"]
                if n not in lists["both"] + lists["colour_only"] and views[n].get("same_colours", True)]
        lists = {**lists, "colour_only": lists["colour_only"] + sorted(set(more), key=more.index)}
    folder = p.folder("02_base")
    out = os.path.join(folder, f"base_seed{s['seed']}.glb")
    command = [PYTHON, os.path.join(SCRIPTS, "generation", "multi_image.py"), *path(lists["both"]),
               "--mode", s["mode"], "--resolution", s["resolution"], "--steps", str(s["steps"]),
               "--faces", str(s["faces"]), "--seed", str(s["seed"]), "--out", out]
    if lists["shape_only"]:
        command += ["--shape-only", *path(lists["shape_only"])]
    if lists["colour_only"]:
        command += ["--colour-only", *path(lists["colour_only"])]
    ok = p.run("02_base", command, [out], {"pictures": lists})
    if ok:
        refs = path(lists["shape_only"] + lists["both"] + lists["colour_only"])
        sheet([("pictures: " + ", ".join(f"{len(lists[k])} {k}" for k in ("both", "shape_only", "colour_only")), refs),
               ("02_base r%d (%s, %s, %d steps)" % (p.round, s["mode"], s["resolution"], s["steps"]),
                render(out, os.path.join(folder, "render")))],
              os.path.join(folder, "compare.jpg"))
    return ok


def node_04_refine(p, part):
    node = f"04_refine_{part}"
    s = settings(node, p.round)
    base = p.newest("02_base", "base_seed*.glb")
    folder = p.folder(node)
    out = os.path.join(folder, f"{part}.glb")
    command = [PYTHON, os.path.join(SCRIPTS, "inpainting", "refine_region.py"), base,
               "--sorted", p.newest("01_sort", "sorted.json"), "--part", part, "--method", s["method"],
               "--margin", str(s["margin"]), "--steps", str(s["steps"]), "--mode", s["mode"], "--seed", str(s["seed"]),
               "--grow", str(s["grow"]), "--out", out]
    if s.get("best_colour") is not None:
        command += ["--best-colour", str(s["best_colour"])]
    if s.get("only"):
        command += ["--only", *s["only"]]
    if s.get("keep_shape"):
        command += ["--keep-shape"]
    if s.get("margin_of_model"):
        command += ["--margin-of-model", str(s["margin_of_model"])]
    if s.get("texture_guidance"):
        command += ["--texture-guidance", str(s["texture_guidance"])]
    if s.get("faces"):
        command += ["--faces", str(s["faces"])]
    ok = p.run(node, command, [out], {"base": base})
    if ok:
        compare_part(p, node, base, out, folder)
    return ok


def compare_part(p, node, before, after, folder):
    """reference pictures | before | after, the same camera on the part's box."""
    info = json.load(open(os.path.join(os.path.splitext(after)[0] + "_work", "refine.json")))
    low, high = info["inner"]
    focus = [*low, *high]
    pictures = [os.path.join(os.path.splitext(after)[0] + "_work", os.path.basename(x[0])) for x in info["pictures"]]
    rows = [(f"{node} r{p.round}: pictures ({info['mode']}; " +
             ", ".join(f"{os.path.basename(x[0]).replace('zoom_picture', 'pic').replace('.png', '')}" for x in info["pictures"]) + ")",
             pictures[:6]),
            ("before (02_base)", render(before, os.path.join(folder, "before"), focus)),
            (f"after ({info['method']}, margin {info['margin']}, {info['steps']} steps)", render(after, os.path.join(folder, "after"), focus))]
    sheet(rows, os.path.join(folder, "compare.jpg"))


def node_05_combine(p):
    s = settings("04_refine", p.round)
    parts = [p.newest(f"04_refine_{part}", f"{part}.glb") for part in s["parts"]]
    parts = [x for x in parts if x]
    base = p.newest("02_base", "base_seed*.glb")
    folder = p.folder("05_combine")
    out = os.path.join(folder, "combined.glb")
    ok = p.run("05_combine", [PYTHON, os.path.join(SCRIPTS, "pipeline", "combine_parts.py"), base, *parts, "--out", out],
               [out], {"parts": parts})
    if ok:
        sheet([("02_base", render(base, os.path.join(folder, "base"))),
               (f"05_combine r{p.round}: base + " + ", ".join(os.path.basename(x)[:-4] for x in parts),
                render(out, os.path.join(folder, "combined")))],
              os.path.join(folder, "compare.jpg"))
    return ok


def node_06_hair(p):
    s = settings("06_hair", p.round)
    model = p.newest("05_combine", "combined.glb") or p.newest("02_base", "base_seed*.glb")
    hair = sorted_result(p).get("hair") or {}
    if not hair.get("hair_colour"):
        print("[06_hair] node 01 found no hair colour: no hair node", flush=True)
        return False
    folder = p.folder("06_hair")
    out = os.path.join(folder, "hair")
    head = p.newest("04_refine_head", "head_work/refine.json")
    focus = [*json.load(open(head))["inner"][0], *json.load(open(head))["inner"][1]] if head else None
    # how far down the hair reaches: node 01's hair box in the front view, as a share of the figure's height
    result = sorted_result(p)
    front = next(v for v in result["views"] if v["name"] == result["lists"]["base"]["front"])
    hair_box = front["part_boxes"].get("hair") or front.get("part_boxes_seen", {}).get("hair")
    share = (hair_box[3] - front["box"][1]) / (front["box"][3] - front["box"][1]) + 0.05 if hair_box else 1.0
    command = [BLENDER, "-b", "--python", os.path.join(SCRIPTS, "pipeline", "hair_strands.py"), "--", model, out,
               "--hair-share", f"{share:.3f}",
               "--hair-colour", hair["hair_colour"], "--count", str(s["count"]), "--length", str(s["length"]),
               "--points", str(s["points"]), "--hue-tolerance", str(s["hue_tolerance"]), "--glb"]
    if focus:
        command += ["--focus", *[f"{v:.5f}" for v in focus]]
    ok = p.run("06_hair", command, [out + ".blend", out + "_front.png"], {"model": model, "hair": hair})
    if ok:
        refs = [os.path.join(os.path.dirname(head), os.path.basename(x[0])) for x in json.load(open(head))["pictures"]][:4] if head else []
        before = render(model, os.path.join(folder, "before"), focus)
        sheet([("06_hair r%d: pictures (hair colour %s from the vision AI)" % (p.round, hair["hair_colour"]), refs),
               ("before: the hair shell (TRELLIS.2)", before),
               ("after: %d strands, Clump + Noise + Profile (Geometry Nodes)" % s["count"],
                [out + f"_{n}.png" for n in ("front", "left", "right", "back")])],
              os.path.join(folder, "compare.jpg"))
    return ok


def node_99_compare(p):
    """overview.jpg: the newest compare.jpg of every node, one under the other; history.jpg: all rounds."""
    folder = os.path.join(p.root, "99_compare")
    os.makedirs(folder, exist_ok=True)
    nodes = sorted(d for d in os.listdir(p.root) if d[:2].isdigit() and d != "99_compare")
    newest = [p.newest(n, "compare.jpg") for n in nodes]
    every = [c for n in nodes for c in sorted(glob.glob(os.path.join(p.root, n, "r*", "compare.jpg")))]
    for name, pics in (("overview.jpg", [x for x in newest if x]), ("history.jpg", every)):
        if not pics:
            continue
        images = [Image.open(x) for x in pics]
        width = max(i.width for i in images)
        out = Image.new("RGB", (width, sum(i.height for i in images)), (40, 40, 44))
        y = 0
        for image in images:
            out.paste(image, (0, y))
            y += image.height
        out.save(os.path.join(folder, name), quality=88)
    lines = ["# %s: all nodes\n" % os.path.basename(p.root)]
    for n in nodes:
        for j in sorted(glob.glob(os.path.join(p.root, n, "r*", "node.json"))):
            info = json.load(open(j))
            lines.append(f"- **{n} r{info['round']}** ({'ok' if info['ok'] else 'failed'}, {info['seconds']} s): "
                         f"{info.get('why') or ''} — settings {json.dumps(info['settings'])}")
    open(os.path.join(folder, "README.md"), "w").write("\n".join(lines) + "\n")
    print(f"[99_compare] {os.path.join(folder, 'overview.jpg')}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Many pictures of one character -> one detailed model (node pipeline)")
    parser.add_argument("pictures", help="the folder with all pictures of the character")
    parser.add_argument("--name", required=True, help="name of the character (outputs/pipeline/<name>)")
    parser.add_argument("--round", type=int, default=max(ROUNDS))
    parser.add_argument("--only", nargs="+", help="run only these nodes (e.g. 04_refine_head 05_combine)")
    parser.add_argument("--compare-only", action="store_true", help="only build the comparison pictures")
    args = parser.parse_args()
    if args.name != ROUNDS_OF:  # a new character: no rounds of another one, only the defaults with the lessons
        ROUNDS.clear()
        ROUNDS[1] = {"why": "first run with the defaults (NODES, with the lessons of " + ROUNDS_OF + ")"}
        args.round = min(args.round, 1) if args.round else 1
    p = Pipeline(os.path.abspath(args.pictures), args.name, args.round)
    wanted = lambda node: not args.only or node in args.only
    if not args.compare_only:
        steps = [("01_sort", lambda: node_01_sort(p)), ("02_base", lambda: node_02_base(p))]
        steps += [(f"04_refine_{part}", lambda part=part: node_04_refine(p, part)) for part in RUN_ORDER]
        steps += [("05_combine", lambda: node_05_combine(p)), ("06_hair", lambda: node_06_hair(p))]
        for node, step in steps:
            if wanted(node):
                step()
                node_99_compare(p)
    node_99_compare(p)


if __name__ == "__main__":
    main()
