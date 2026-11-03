"""Node 01: the vision AI looks at every picture of one character and the rules turn its answers into
picture lists for the other nodes (which picture goes into which stage of which node).

The AI (Qwen3-VL, as in organise_pictures.py) answers; nothing here knows what the pictures show.

  1. every picture     how many views it shows (a turnaround sheet has several), a box around each view
  2. every view        cut out, then looked at alone: colour or grey sculpt, which side, whole figure or a
                       close-up, which body parts it shows, big wings on the back yes / no, a scene around it
  3. every part        a box around each body part the view shows (head, hair, chest, legs, feet, back)
  4. check             every part box cut out and looked at alone: is it really that part of the main character
                       (not of a prop or another creature), colour or grey; every colour close-up next to the
                       whole colour view: the same outfit colours? (if not: shape only, e.g. another material)
  5. rules             the lists (see choose_lists): which views build the shape, which only give colours

Writes into --out: sorted.json (all answers + the lists), views/*.png (the cut-out views) and
contact.jpg (all views with the AI's tags, to check it at a glance).

Usage (TRELLIS.2 environment):
    python scripts/pipeline/sort_pictures.py "pictures/local model" --out outputs/pipeline/morrigan/01_sort
"""
import _paths  # noqa: F401  (the scripts of the other folders)

import argparse
import glob
import json
import os
import time

from PIL import Image, ImageDraw

import organise_pictures as ai

SIDES = ai.SIDES
# the parts that get their own refine node: small boxes get more zoom (the user: bust, belly and butt
# looked bad inside the big chest and legs boxes, and the butt sat on the seam between back and legs)
PARTS = ("head", "hair", "bust", "belly", "back", "butt", "thighs", "lower_legs", "feet")
PART_WORDS = {  # what the AI is asked to put a box around
    "head": "the head (face and the hair on top of the head, from the top of the head down to the chin)",
    "hair": "all of the hair, from the top of the head down to the tips of the longest strands",
    "bust": "the breasts (from the collarbones down to just below the breasts)",
    "belly": "the belly and waist (from just below the breasts down to the hips)",
    "back": "the upper back (from the shoulders down to the waist)",
    "butt": "the hips and the bottom (buttocks)",
    "thighs": "the thighs (both upper legs, from the hips down to the knees)",
    "lower_legs": "the lower legs (from the knees down to the ankles)",
    "feet": "the feet and shoes",
}

PART_SIDE = {"back": "back", "butt": "back"}  # the side a part is seen best from (all others: the front)


def measured_colours(image):
    """Grey sculpt render or colour picture, measured (no AI): grey renders have almost no colour saturation."""
    import numpy as np
    rgb = np.asarray(ai.shrink(image, 256).convert("RGB"), np.float32) / 255
    saturation = (rgb.max(-1) - rgb.min(-1)) / (rgb.max(-1) + 1e-6)
    return ("grey" if np.percentile(saturation, 75) < 0.2 else "colour"), round(float(np.percentile(saturation, 75)), 3)

VIEWS = """Look at this picture. It may show one figure, the same figure several times from different sides next to
each other (a turnaround sheet), or only a close-up of a part.
Answer only with JSON, no other text:
{"object": "a short name of the main character",
 "views": [{"box": [x0, y0, x1, y1]}]   // one box around each view of the character, coordinates from 0 to 1000
}"""

VIEW = """This is one view of: %s.
Answer only with JSON, no other text:
{"colours": "colour" | "grey",          // grey = an uncoloured sculpt / clay render
 "side": one of %s,                     // which side of the character we see
 "whole": true | false,                 // the whole character from head to feet (false for a close-up)
 "parts": [the body parts that can be seen well, from: %s],
 "wings_on_back": true | false,         // big wings growing from the back (small wings on the head do not count)
 "pose": "neutral" | "action",          // neutral = standing straight (A- or T-pose, arms down or out); action = any other pose
 "scene": [things that are not part of the character: "pedestal", "rail", "candles", "pumpkin", "bag", "floor", ...],
 "why": "a few words"}"""

BOX = """Put a box around %s of the character in this picture. Only that part.
Answer only with JSON, no other text: {"box": [x0, y0, x1, y1]}   // coordinates from 0 to 1000 of the picture"""

CHECK = """This is a cut-out of a picture of a character.
Answer only with JSON, no other text:
{"is_part": true | false,        // the cut-out shows %s (other parts around it are fine)
 "main_character": true | false, // of the main character herself / himself (not of a prop, a pet or another creature)
 "colours": "colour" | "grey",   // grey = an uncoloured sculpt / clay render
 "why": "a few words"}"""

SAME_COLOURS = """The first picture shows the whole character in colour. The second is a close-up.
Compare the colours of the clothes, skin and hair that both show.
Answer only with JSON, no other text:
{"same_colours": true | false,   // the close-up has the same colours as the first picture (only seen closer)
 "why": "a few words"}"""

HAIR = """What colour is the character's hair in this picture?
Answer only with JSON, no other text: {"colour": one of %s}"""
HUES = {"red": 0, "orange": 30, "brown": 25, "blonde": 45, "yellow": 55, "green": 125, "teal": 170, "cyan": 185,
        "blue": 220, "purple": 275, "pink": 320}  # degrees on the colour circle; the exact colour is measured

SAME_CHARACTER = """The first picture shows the main character. Is the second picture the same character (maybe seen
from another side, closer, or as an uncoloured grey sculpt)? A pet, a prop or another creature is not the same.
Answer only with JSON, no other text: {"same_character": true | false, "why": "a few words"}"""

SIDE_ORDER = ("front", "front-left", "front-right", "left", "right", "back-left", "back-right", "back", "top", "bottom", "unknown")


def look(model, processor, folder, out, size):
    """Steps 1-3: the views of every picture, their tags and the boxes of their parts."""
    files = sorted(f for f in glob.glob(os.path.join(folder, "*")) if f.lower().endswith(ai.PICTURE_TYPES))
    views, thing = [], None
    for path in files:
        image = Image.open(path).convert("RGB")
        shown = ai.shrink(image, size)
        answer, raw = ai.ask(model, processor, [shown], VIEWS, 600)
        thing = thing or answer.get("object")
        boxes = [ai.to_pixels(v.get("box"), image.width, image.height, shown.size)
                 for v in answer.get("views", []) if isinstance(v, dict)]
        boxes = [b for b in boxes if b] or [(0, 0, image.width, image.height)]
        print(f"  {os.path.basename(path)}: {len(boxes)} view(s)", flush=True)
        for k, box in enumerate(boxes):
            name = os.path.splitext(os.path.basename(path))[0] + (f"_view{k + 1}" if len(boxes) > 1 else "")
            crop = image.crop(box)
            view_path = os.path.join(out, "views", name + ".png")
            crop.save(view_path)
            shown_crop = ai.shrink(crop, size)
            tags, _ = ai.ask(model, processor, [shown_crop], VIEW % (thing or "a character", "|".join(f'"{s}"' for s in SIDES),
                                                                    ", ".join(f'"{p}"' for p in PARTS)), 400)
            parts = [p for p in tags.get("parts", []) if p in PARTS] if isinstance(tags.get("parts"), list) else []
            boxes_of_parts = {}
            for part in parts:
                answer, _ = ai.ask(model, processor, [shown_crop], BOX % PART_WORDS[part], 100)
                part_box = ai.to_pixels(answer.get("box"), crop.width, crop.height, shown_crop.size)
                if part_box:  # in pixels of the original picture
                    boxes_of_parts[part] = [part_box[0] + box[0], part_box[1] + box[1], part_box[2] + box[0], part_box[3] + box[1]]
            colours, saturation = measured_colours(crop)
            if colours != tags.get("colours"):
                print(f"    {name}: the AI said {tags.get('colours')}, measured {colours} (saturation {saturation})", flush=True)
            view = {"name": name, "picture": os.path.abspath(path), "view": os.path.relpath(view_path, out),
                    "box": list(box), "colours": colours, "colours_ai": tags.get("colours"), "saturation": saturation, "side": tags.get("side") if tags.get("side") in SIDES else "unknown",
                    "whole": tags.get("whole") is True, "parts": parts, "wings_on_back": tags.get("wings_on_back") is True,
                    "pose": tags.get("pose") if tags.get("pose") in ("neutral", "action") else "action",
                    "scene": tags.get("scene") if isinstance(tags.get("scene"), list) else [], "why": tags.get("why"),
                    "part_boxes": boxes_of_parts}
            print(f"    {name}: {view['colours']}, {view['side']}, whole {view['whole']}, parts {parts}, "
                  f"back wings {view['wings_on_back']}, scene {view['scene']}", flush=True)
            views.append(view)
    return thing, views


def check(model, processor, thing, views, out, size):
    """Step 4: every view next to a reference view (the same character?), every part box alone, and every
    colour close-up next to the whole colour view (the same colours?)."""
    for v in views:  # the first look's answers stay as they were, so the check can run again
        for key in ("part_boxes", "whole", "colours"):
            v.setdefault(key + "_seen", v[key])
            v[key] = dict(v[key + "_seen"]) if key == "part_boxes" else v[key + "_seen"]
    whole = [v for v in views if v["whole"]] or views

    def reference_for(v):
        """The whole view that shows the most parts, of the same kind (grey or colour) if there is one: an
        uncoloured sculpt next to a coloured picture made the AI answer 'other hair colour, not the same'."""
        same_kind = [w for w in whole if w["colours"] == v["colours"]] or whole
        return max(same_kind, key=lambda w: len(w["part_boxes"]))

    shown = lambda v: ai.shrink(Image.open(os.path.join(out, v["view"])).convert("RGB"), 512)
    for v in views:
        v["same_character"] = True
        reference = reference_for(v)
        if v is not reference:
            answer, _ = ai.ask(model, processor, [shown(reference), shown(v)], SAME_CHARACTER, 150)
            v["same_character"] = answer.get("same_character") is not False
            v["character_check"] = answer
            if not v["same_character"]:
                print(f"    {v['name']}: not the main character ({answer.get('why')})", flush=True)
                v["part_boxes"], v["whole"] = {}, False
    whole_colour = next((v for v in views if v["whole"] and v["colours"] == "colour"), None)
    for v in views:
        picture = Image.open(v["picture"]).convert("RGB")
        v["part_checks"] = {}
        for part, box in list(v["part_boxes"].items()):
            crop = ai.shrink(picture.crop(box), 512)
            answer, _ = ai.ask(model, processor, [crop], CHECK % PART_WORDS[part], 200)
            v["part_checks"][part] = answer
            if answer.get("is_part") is not True or answer.get("main_character") is not True:
                del v["part_boxes"][part]
                print(f"    {v['name']}: no {part} ({answer.get('why')})", flush=True)
        # the colours stay measured (saturation), the AI's looks are only kept for the record
        v["same_colours"] = True
        if v["colours"] == "colour" and not v["whole"] and whole_colour:
            pair = [ai.shrink(Image.open(os.path.join(out, x["view"])).convert("RGB"), 512) for x in (whole_colour, v)]
            answer, _ = ai.ask(model, processor, pair, SAME_COLOURS, 200)
            v["same_colours"] = answer.get("same_colours") is not False
            v["colour_check"] = answer
        print(f"    {v['name']}: {v['colours']}, parts {list(v['part_boxes'])}, same colours {v['same_colours']}", flush=True)
    return views


def hair_colour(model, processor, views, out):
    """The hair colour (for node 06: which faces of the model are hair). The AI names it in a colour close-up
    of the head; the exact colour is then measured: the middle of the crop's pixels of that hue."""
    import numpy as np
    for v in sorted(views, key=lambda v: v["whole"]):  # close-ups first: more hair pixels
        box = v["part_boxes"].get("hair") or v["part_boxes"].get("head")
        if v["colours"] != "colour" or not box or not v.get("same_colours", True):
            continue
        crop = Image.open(v["picture"]).convert("RGB").crop(box)
        answer, _ = ai.ask(model, processor, [ai.shrink(crop, 512)], HAIR % ", ".join(f'"{h}"' for h in HUES), 60)
        name = answer.get("colour")
        if name not in HUES:
            continue
        rgb = np.asarray(ai.shrink(crop, 256), np.float32).reshape(-1, 3) / 255
        mx, mn = rgb.max(1), rgb.min(1)
        d = mx - mn + 1e-8
        r, g, b = rgb.T
        hue = np.where(mx == r, ((g - b) / d) % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) * 60
        near = np.minimum(abs(hue - HUES[name]), 360 - abs(hue - HUES[name])) < 35
        pick = near & ((mx - mn) / (mx + 1e-8) > 0.25) & (mx > 0.1)
        if pick.sum() < 100:
            continue
        colour = np.median(rgb[pick], 0)
        result = {"hair_colour": "#%02x%02x%02x" % tuple(int(c * 255) for c in colour), "name": name,
                  "from": v["name"], "pixels": int(pick.sum())}
        print(f"    hair colour: {result}", flush=True)
        return result
    return None


def choose_lists(views):
    """Step 4, the rules (from the night tests, outputs/tests/2026-11-03/ROUNDS.md):
    - the shape (stages 1 + 2) only from views of one and the same version: the version most whole views show
      (wings on the back or not); a tie goes to the version of the grey views (clean sculpt renders)
    - colour views of another version only give colours (colour only); grey views never give colours
    - close-ups never build the whole figure, only the refine node of their part
    - a colour close-up with other colours than the whole colour view (another material) is shape only"""
    colour = lambda v: v["colours"] == "colour" and v.get("same_colours", True)
    # a version = which parts (wings on the back or not) AND which pose: an action pose mixed into A-pose views
    # ruined models in the night tests. The version most whole views show wins; a tie goes to the grey views'
    whole = [v for v in views if v["whole"]]
    key = lambda v: (v["wings_on_back"], v.get("pose", "action"))
    counts = {}
    for v in whole:
        counts[key(v)] = counts.get(key(v), 0) + 1 + 0.5 * (v["colours"] == "grey")
    version = max(counts, key=counts.get) if counts else (False, "neutral")
    by_side = lambda v: SIDE_ORDER.index(v["side"]) if v["side"] in SIDE_ORDER else len(SIDE_ORDER)
    roles = lambda vs: {
        "both": [v["name"] for v in sorted(vs, key=by_side) if colour(v) and key(v) == version],
        "colour_only": [v["name"] for v in sorted(vs, key=by_side) if colour(v) and key(v) != version],
        "shape_only": [v["name"] for v in sorted(vs, key=by_side) if not colour(v) and key(v) == version],
    }
    lists = {"version": {"wings_on_back": version[0], "pose": version[1]}, "base": roles(whole)}
    shape = sorted([v for v in whole if key(v) == version], key=by_side)
    lists["base"]["front"] = shape[0]["name"] if shape else None  # the view the model's camera is fitted to
    for part in PARTS:
        lists[part] = roles([v for v in views if part in v["part_boxes"]])
        lists[part]["side"] = PART_SIDE.get(part, "front")
    used = {name for key in ("base",) + PARTS for role in ("both", "colour_only", "shape_only") for name in lists[key][role]}
    lists["left_out"] = [v["name"] for v in views if v["name"] not in used]
    return lists


def contact_sheet(views, out, cell=360):
    """All views in one picture, each with the AI's tags, to check the AI at a glance."""
    cols = 5
    rows = -(-len(views) // cols)
    sheet = Image.new("RGB", (cols * cell, rows * (cell + 54)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, v in enumerate(views):
        image = Image.open(os.path.join(out, v["view"])).convert("RGB")
        image.thumbnail((cell, cell))
        x, y = (i % cols) * cell, (i // cols) * (cell + 54)
        sheet.paste(image, (x + (cell - image.width) // 2, y))
        lines = [v["name"].replace("deyvson-william-", "")[:44],
                 f"{v['colours']}, {v['side']}, {'whole' if v['whole'] else 'close-up'}{', BACK WINGS' if v['wings_on_back'] else ''}",
                 ", ".join(v["parts"])]
        for k, line in enumerate(lines):
            draw.text((x + 4, y + cell + 2 + 16 * k), line, fill="black")
    sheet.save(os.path.join(out, "contact.jpg"), quality=90)


def main():
    parser = argparse.ArgumentParser(description="Node 01: the vision AI sorts a character's pictures into lists")
    parser.add_argument("folder", help="all pictures of one character")
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default=ai.MODELS[0], choices=ai.MODELS)
    parser.add_argument("--size", type=int, default=1024, help="longest side of the pictures the AI sees")
    parser.add_argument("--rules-only", action="store_true", help="keep the AI's answers in sorted.json, only apply the rules again")
    parser.add_argument("--check-again", action="store_true", help="keep steps 1-3 in sorted.json, run the check (step 4) again")
    args = parser.parse_args()
    os.makedirs(os.path.join(args.out, "views"), exist_ok=True)
    start = time.time()
    result_path = os.path.join(args.out, "sorted.json")
    if args.rules_only:
        result = json.load(open(result_path))
    else:
        model, processor = ai.load(args.model, cpu=False)
        if args.check_again:
            result = json.load(open(result_path))
        else:
            thing, views = look(model, processor, args.folder, args.out, args.size)
            result = {"model": args.model, "object": thing, "views": views}
        print("Checking every part and every colour close-up", flush=True)
        check(model, processor, result["object"], result["views"], args.out, args.size)
        result["hair"] = hair_colour(model, processor, result["views"], args.out)
    result["lists"] = choose_lists(result["views"])
    result["seconds"] = round(time.time() - start)
    json.dump(result, open(result_path, "w"), indent=1)
    contact_sheet(result["views"], args.out)
    print(json.dumps(result["lists"], indent=1), flush=True)
    print(f"Result: {result_path}", flush=True)
    print(f"Done in {time.time() - start:.0f} s", flush=True)


if __name__ == "__main__":
    main()
