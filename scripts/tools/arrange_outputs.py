"""outputs/<object>/: one folder per object, everything of it inside.

    outputs/robot_bee/v01_base_one_view_in0_view1_1536_seed42.glb     its models as versions, the title says how
                      v02_corrected_round1_kept_3_of_4_parts.glb      each was made (PICKED = the seed judge's pick)
                      versions.md      every version: its title, when it was made, where it came from
                      work/            what the programs need (the flows' node folders, the correction loop's steps)

The models in work/ are put next to it as versions by hard links (the same file under a second name, no extra room).

  1. the results downloaded from the uni server (outputs/cloud/*.glb, named <job>_<date>__<file>) are MOVED into
     their object's folder. Jobs sent from Blender are called geometry_0 or model: they go to _unnamed/<name>
  2. every flow's models (outputs/<flow>/work/02_base, 04_inpaint_*) are linked, the seed judge's pick marked PICKED

    python scripts/tools/arrange_outputs.py            # shows what it would do
    python scripts/tools/arrange_outputs.py --do       # does it (run again any time: what is there is left alone)
"""
import argparse
import glob
import json
import os
import re
import shutil
import time

PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(PROJECT, "outputs")
MODELS = OUT
DATE = r"_?20\d{6}-\d{6}"
UNNAMED = ("geometry", "model", "view_", "picture", "in0_", "in1_", "base")  # names that don't say what the object is


# the objects worked on so far: a run called bee_crown_plain or r_head3_room_prot belongs to the bee and to head3
OBJECTS = ("robot_hand_1", "robot_hand_2", "robot_hand_3", "robot_bee", "robot_fly", "simple_robot", "metal_robot",
           "super_woman", "video_character_ex", "nefertiti", "valkyrie", "wanderer", "monster", "dragon", "knight",
           "raven", "ninja", "head3", "girl", "bee", "chest", "crown", "wheel", "torso", "gate", "helmet", "robot")
TESTS = ("addon", "server_test", "multi_diffusion", "best_6nov")  # runs that tried the tools, not an object


def object_of(name):
    """The object's folder and what is left of the run's name (it goes into the version's title)."""
    name = re.sub(DATE + ".*", "", name)
    name = re.sub(r"_(generate|job|pipeline)$", "", name)
    short = re.sub(r"^r_", "", name)
    for known in OBJECTS:
        if re.match(rf"{known}([_-]|$)", short):
            return known, short[len(known):].strip("_-")
    for known in TESTS:
        if short.startswith(known):
            return os.path.join("_tests", known), short[len(known):].strip("_-")
    if name.startswith(UNNAMED):
        return os.path.join("_unnamed", name), ""
    return name, ""


def clean(text):
    text = re.sub(DATE, "", text)
    text = re.sub(r"picture0_?|_steps25|multidiffusion_?|_generate|_job|_pipeline|__live_|ali-jafari-", "_", text)
    return re.sub(r"[^A-Za-z0-9+.-]+", "_", text).strip("_") or "model"


def from_cloud():
    """(object, title, file, 'move') of every downloaded result."""
    found = []
    for path in sorted(glob.glob(os.path.join(OUT, "cloud", "*.glb"))):
        parts = os.path.basename(path)[:-4].split("__")  # <job>_<date>__<file>, or <object>__<run's title>_<date>__<file>
        thing, extra = object_of(parts[0])
        middle = clean(parts[1]) if len(parts) > 2 else ""
        result = clean(parts[-1].replace(re.sub(DATE + ".*", "", parts[0]), ""))
        title = "_".join(piece for piece in (extra, middle, result) if piece and piece != "model") or "model"
        found.append((thing, title, path, "move"))
    return found


def base_title(folder):
    settings = json.load(open(os.path.join(folder, "settings.json")))
    views = settings.get("views") or []
    one = ""
    if len(views) == 1:  # which picture: the data sheet knows where the view came from
        try:
            sheet = json.load(open(os.path.join(views[0][0].rsplit("/views/", 1)[0], "sheet.json")))
            source = next(v["from"] for v in sheet["views"] if v["name"] == os.path.basename(views[0][0])[:-4])
            one = "_" + clean(os.path.splitext(os.path.basename(source))[0])
        except (OSError, StopIteration, KeyError, ValueError):
            pass
    kind = f"one_view{one}" if len(views) == 1 else f"{len(views)}_views_mixed"
    return (f"base_{kind}_{settings.get('resolution', '1024')}_seed{settings.get('seed', 42)}"
            + ("_colours_from_main" if settings.get("colours") == "main" else "")
            + ("_server" if settings.get("where") == "server" else ""))


def from_flows():
    found = []
    for flow in sorted(glob.glob(os.path.join(OUT, "*", "work", ""))):
        thing, extra = os.path.basename(os.path.dirname(flow.rstrip("/"))), ""  # the flow's name is the object's folder
        picked = set()
        for pick in glob.glob(os.path.join(flow, "02b_pick_seed", "*", "pick.json")):
            newest = max(glob.glob(os.path.join(flow, "02b_pick_seed", "*", "pick.json")), key=os.path.getmtime)
            if pick == newest:
                picked.add(os.path.realpath(json.load(open(pick))["best"]))
        for folder in sorted(glob.glob(os.path.join(flow, "02_base", "*", ""))):
            for path in glob.glob(os.path.join(folder, "*.glb")):
                if os.path.isfile(os.path.join(folder, "settings.json")):
                    title = (extra + "_" if extra else "") + base_title(folder) + ("_PICKED" if os.path.realpath(path) in picked else "")
                    found.append((thing, title, path, "link"))
        for folder in sorted(glob.glob(os.path.join(flow, "04_inpaint_*", "*", ""))):
            step = os.path.basename(os.path.dirname(folder.rstrip("/"))).replace("04_", "")
            for path in sorted(glob.glob(os.path.join(folder, "*.glb"))):
                found.append((thing, (extra + "_" if extra else "") + f"{step}_{clean(os.path.basename(path)[:-4])}", path, "link"))
        for path in sorted(glob.glob(os.path.join(flow, "06_finish", "*", "*.glb"))):  # the new part in the whole model
            found.append((thing, (extra + "_" if extra else "") + "FINISHED_part_in_the_whole_model", path, "link"))
    return found


def main():
    parser = argparse.ArgumentParser(description="outputs/<object>: one folder per object, its models as versions")
    parser.add_argument("--do", action="store_true", help="really move and link (without it: only shows the plan)")
    parser.add_argument("--more", nargs=3, action="append", metavar=("OBJECT", "TITLE", "FILE"), default=[],
                        help="one more model to put there (linked), e.g. a result of the Blender add-on")
    args = parser.parse_args()
    found = from_cloud() + from_flows() + [(o, clean(t), os.path.join(PROJECT, f) if not os.path.isabs(f) else f, "link") for o, t, f in args.more]
    by_thing = {}
    for thing, title, path, how in found:
        by_thing.setdefault(thing, []).append((os.path.getmtime(path), title, path, how))
    moved = linked = 0
    planned = {"move": 0, "link": 0}
    for thing, models in sorted(by_thing.items()):
        folder = os.path.join(MODELS, thing)
        record = os.path.join(folder, "versions.json")
        known = json.load(open(record)) if os.path.isfile(record) else []  # [{"version", "title", "made", "from"}]
        here = {entry["from"] for entry in known}
        new = [m for m in sorted(models) if os.path.relpath(m[2], PROJECT) not in here]
        print(f"{thing}: {len(known)} there, {len(new)} new")
        for made, title, path, how in new:
            number = len(known) + 1
            planned[how] += 1
            name = f"v{number:02d}_{title}{os.path.splitext(path)[1].lower() or '.glb'}"  # a .blend stays a .blend
            print(f"    {how:4}  {name}   <- {os.path.relpath(path, OUT)}")
            if args.do:
                os.makedirs(folder, exist_ok=True)
                target = os.path.join(folder, name)
                if how == "move":
                    shutil.move(path, target)
                    moved += 1
                else:
                    try:
                        os.link(path, target)
                    except OSError:  # a disk without hard links: a copy
                        shutil.copy2(path, target)
                    linked += 1
            known.append({"version": number, "file": name, "title": title, "made": time.strftime("%Y-%m-%d %H:%M", time.localtime(made)),
                          "from": os.path.relpath(path, PROJECT)})
        if args.do and new:
            json.dump(known, open(record, "w"), indent=1)
            with open(os.path.join(folder, "versions.md"), "w") as f:
                f.write(f"# {thing}\n\n| Version | Title | Made | Came from |\n|---|---|---|---|\n")
                for entry in known:
                    f.write(f"| v{entry['version']:02d} | {entry['title']} | {entry['made']} | `{entry['from']}` |\n")
    print(f"moved {moved}, linked {linked}" if args.do else f"would move {planned['move']}, would link {planned['link']}")
    if not args.do:
        print("Nothing was changed: add --do")


if __name__ == "__main__":
    main()
