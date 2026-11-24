"""The Slicer's plan for any character: Qwen3-VL looks at the picture(s) and says what is there; the parts and
their prompts are then written from one fixed template, so every project has the same locks.

Always: 00 the whole character in an A-pose (the base), 01 the centre (the torso without head, arms and legs),
then the parts Qwen3-VL finds (head, hair, arms, legs, loose clothes, props), each made alone.

    python scripts/pipeline/slicer_plan.py outputs/slicer/<name>      # reads its sources/, writes its slicer.json
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "generation"))
import _paths  # noqa: F401,E402

PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_HUB_CACHE", os.path.join(PROJECT, "local_ai_models"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from PIL import Image  # noqa: E402

# the same for every project and every part
LOCKED = {"pose": "A-pose", "size": "2048 x 2048", "background": "#808080 grey", "views": "front, back"}
LOCK = (f"Locked: one picture {LOCKED['size']}, the part centred and whole (nothing cut by the edge), plain flat "
        "#808080 grey background, soft even studio light, no shadow on the floor, no text.")
GREY = ("The same picture as an unpainted grey clay sculpt: every shape, fold and detail kept exactly, one even grey "
        "material, no colours, no patterns.")
EDITOR = "Qwen-Image-Edit 2511 (local, ComfyUI), made at about 1024 and brought to 2048 by Real-ESRGAN"

QUESTION = """You prepare this character for 3D modelling, one part at a time. Look at the picture(s) and answer
with JSON only:
{"name": "a short name for the character, 2 to 4 words",
 "keep": "one short phrase: what must stay exactly the same (face, hair, the outfit's main pieces)",
 "props": ["items held or separate from the body: weapons, bags, shields ...; [] if none"],
 "center": "what the torso shows without head, arms and legs: the clothes on the chest, waist and neck",
 "parts": [{"name": "a short name", "show": "what the picture of this part alone shows, e.g. 'her long red hair as a wig, hanging straight down'"}]}
Rules for "parts", in this order, each only if the character has it:
1. "head": the head with what is worn on it (helmet, ears, crown); short hair stays with the head.
2. "hair": long or big hair alone, as a wig.
3. "arms": both arms together in one picture, with gloves, sleeves, bracelets.
4. "legs": both legs together in one picture, with the shoes or boots.
5. each loose piece of clothing that hangs free: a cape, a long skirt, a coat tail.
6. each prop, one part each.
Never a part for the torso, the upper body or the chest: that is "center". Never left and right as two parts."""


def the(text):
    """'head with ears' -> 'the head with ears' (a phrase that already starts with a/an/the/her/his stays)."""
    text = text.strip().rstrip(".")
    return text if re.match(r"(?i)(a|an|the|her|his|their|its|both)\b", text) else "the " + text


def slug(text):
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:24] or "part"


def plan_from(answer, name, sources):
    keep = answer.get("keep") or "the face, the hair and the outfit"
    props = [p for p in answer.get("props") or [] if isinstance(p, str) and p.strip()]
    leave = f" Leave out {', '.join(props)}." if props else ""
    parts = [
        {"id": "00_apose", "name": "Whole character, A-pose",
         "what": "the base all slices come from: alone, standing straight, arms down and out at about 45 degrees, "
                 "legs a little apart, front view" + (", without " + ", ".join(props) if props else ""),
         "prompt": "Show this character alone in an A-pose: standing straight, facing the camera, arms down and away "
                   "from the body at about 45 degrees, palms toward the body, legs slightly apart, feet flat. "
                   f"Keep {keep} exactly as they are.{leave} {LOCK}"},
        {"id": "01_center", "name": "Centre: torso",
         "what": f"always first: {answer.get('center') or 'the torso'}; no head, no arms, no legs",
         "prompt": f"Show only the torso: {the(answer.get('center') or 'clothes on the chest, waist and neck')}, as on "
                   "a tailor's dummy, front view, upright. No head, no arms, no legs. Keep the fabric, its seams and "
                   f"details exactly. {LOCK}"},
    ]
    used = {"apose", "center"}
    found = [p for p in answer.get("parts") or [] if isinstance(p, dict)]
    named = " ".join(str(p.get("name", "")) + " " + str(p.get("show", "")) for p in found).lower()
    for prop in props:  # every prop is a part of its own (the model sometimes lists them only as props)
        if prop.lower() not in named:
            found.append({"name": prop, "show": f"the {prop} alone, whole, no hands"})
    for i, part in enumerate(found):
        if not isinstance(part, dict) or not part.get("show"):
            continue
        label = str(part.get("name") or "").strip()
        if not label or label.lower() in ("prop", "props", "item", "weapon part"):
            label = " ".join(re.sub(r"(?i)^(a|an|the)\s+", "", part["show"]).split()[:3])
        s = slug(label)
        if s in used or any(w in s for w in ("torso", "upper_body", "chest", "center", "centre")):
            continue
        used.add(s)
        parts.append({"id": f"{len(parts):02d}_{s}", "name": label.capitalize(),
                      "what": part["show"],
                      "prompt": f"Show only {the(part['show'])}, nothing else, seen from the front. Keep its shape, "
                                f"colours and details exactly. {LOCK}"})
        if len(parts) >= 10:
            break
    for p in parts:
        p["grey_prompt"] = GREY
    return {"name": name, "character": answer.get("name", ""), "sources": sources, "source": sources[0],
            "locked": LOCKED, "editor": EDITOR,
            "instruction": "Slice this character into parts that are best made alone in 3D. First the whole "
                           "character in an A-pose (the base), then always the centre (the torso without head, arms "
                           "and legs), then the rest one by one. For every part: one coloured picture and the same "
                           f"as grey clay, both locked to {LOCKED['size']} on a flat grey background.",
            "parts": parts}


def make_plan(folder, cpu=False):
    from organise_pictures import ask, load, shrink
    from describe_colours import on_grey
    plan_file = os.path.join(folder, "slicer.json")
    old = json.load(open(plan_file)) if os.path.isfile(plan_file) else {}
    sources = old.get("sources") or sorted(
        os.path.relpath(os.path.join(folder, "sources", f), PROJECT) for f in os.listdir(os.path.join(folder, "sources")))
    model, processor = load("Qwen/Qwen3-VL-8B-Instruct", cpu)
    images = [shrink(on_grey(Image.open(os.path.join(PROJECT, s))), 768) for s in sources[:4]]
    answer, raw = ask(model, processor, images, QUESTION, max_tokens=900)
    if not answer.get("parts"):
        raise SystemExit(f"Qwen3-VL gave no parts: {raw[:500]}")
    plan = plan_from(answer, os.path.basename(folder.rstrip("/")), sources)
    plan["answer"] = answer  # what the model said, kept to see why the parts are what they are
    json.dump(plan, open(plan_file + ".tmp", "w"), indent=1, ensure_ascii=False)
    os.replace(plan_file + ".tmp", plan_file)
    return plan


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("folder", help="outputs/slicer/<name> with its pictures in sources/")
    p.add_argument("--cpu", action="store_true")
    a = p.parse_args()
    plan = make_plan(a.folder, a.cpu)
    for part in plan["parts"]:
        print(f"{part['id']:<22} {part['what'][:100]}")


if __name__ == "__main__":
    main()
