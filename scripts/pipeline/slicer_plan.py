"""The Slicer's plan for any character: Qwen3-VL looks at the picture(s) and says what is there; the parts and
their prompts are then written from one fixed template, so every project has the same locks.

Always: 00 the whole character in an A-pose, 01 the base body (the same body without its outfit and accessories, in
a matte grey shop mannequin: what every piece is put on), then every piece Qwen3-VL finds made alone: hair, what is worn on the
head, each piece of clothing (empty, as on an invisible mannequin), gloves, shoes, accessories, props.

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
# the base body's clay keeps the doll rule ("every detail" alone made Lou's a muscle-anatomy sculpt with nipples)
GREY_BASE = ("The same picture as an unpainted grey clay sculpt of a smooth featureless doll body: the same soft "
             "shapes and outline exactly, smooth surfaces, no muscle lines, no nipples, no genitals, nothing sexual, "
             "one even grey material, no colours, no patterns.")
EDITOR = "Qwen-Image-Edit 2511 (local, ComfyUI), made at about 1024 and brought to 2048 by Real-ESRGAN"

# the layers made from the A-pose keep its frame (no new centring): the base body and every piece line up with it
FRAME = "Plain flat #808080 grey background, soft even studio light, no shadow on the floor, no text."
# a piece alone stays where it is on the A-pose, at its size (Nov 24: redrawn large and centred, the pieces no longer
# fitted the body: "they all have to be consistent together"; "invisible mannequin" alone drew a whole mannequin)
ALONE = ("exactly where it is in the picture, at exactly the same size, position and shape, as if worn by an invisible "
         "body: hollow and empty, with empty openings where the body would be. Remove the body, the head, the skin and "
         "every other item. Keep its colours and every detail exactly")
# the base body: the A-pose's own body as a matte grey shop mannequin (a 3D base mesh). On skin the edit model drew
# nipples whatever the words said (ryan, Lou, Nov 24: 6 of 6 failed the check); as a plastic mannequin it has no
# anatomy by nature (3 of 3 passed). Asked as "the same character as a base body" it drew a new, slimmer woman.
BASE = ("Turn the character into a smooth matte light-grey plastic shop-window mannequin with exactly the same body: "
        "remove all the clothes, the hair and all accessories (headwear, gloves, cuffs, jewellery, weapons and the "
        "shoes). The mannequin has the exact face shape, the exact body shape with the same width of shoulders, chest, "
        "waist, hips, bottom and thighs, the same height, the same A-pose, the same size and position in the picture "
        "and the same camera. The chest, the bottom, the hips and the legs keep exactly their size and shape, even when "
        "they are big: 1 to 1 the same outline. Like every shop mannequin it has no anatomical details: a smooth plain chest, rounded without "
        "any points or bumps at the tips, a smooth plain crotch, a smooth bald head, one even matte grey plastic "
        "everywhere.")

# the fallback when the doll body comes out with nipples or genitals twice (the edit model doesn't always obey)
BASE_UNDERWEAR = ("Change only the clothes, the accessories and the hairstyle, nothing else: replace the outfit with "
                  "plain tight-fitting neutral grey underwear (a sports bra and briefs), remove every accessory, glove, "
                  "cuff, headwear, jewellery, weapon and the shoes (barefoot), and pull the hair back tightly into a "
                  "small bun. Keep everything else exactly as it is in the picture: the same face, the same skin, the "
                  "same body shape with the same width of shoulders, chest, waist, hips and thighs, the same height, "
                  "the same A-pose, the same size and position in the picture and the same camera.")

QUESTION = """You prepare this character for 3D modelling in layers, from the inside to the outside: first the bare
body, then every piece made alone and put on it. Look at the picture(s) and answer with JSON only:
{"name": "a short name for the character, 2 to 4 words",
 "keep": "one short phrase: what must stay exactly the same (face, hair, the outfit's main pieces)",
 "props": ["items held in the hands: weapons, glasses, bags, shields ...; [] if none"],
 "parts": [{"name": "a short name", "show": "what this piece alone is, with its colours, e.g. 'the black leather corset with a heart cut-out'", "category": "..."}]}
One part for each separate thing the character wears or has on top of the bare body. "category" is one of:
"clothing" (bodysuit, corset, top, shirt, dress, skirt, trousers, leggings, shorts), "jewellery" (necklace, choker,
belt, collar, piercings), "outerwear" (cape, coat, jacket, cloak), "footwear" (shoes, boots, sandals: the pair in one
part), "handwear" (gloves, sleeves, cuffs, bracelets, arm bands: a pair in one part), "hand item" (what is held in a
hand), "hair" (the hair), "headwear" (anything worn on the head: hat, crown, headband, ears, horns, small wings on
the head), "attachment" (wings on the back, a tail). Never a part for the body or for body parts (head, face, arms,
legs, torso): the body is made separately."""

COMPLETE_TEXT = ("This picture shows {piece} cut out from a character: the parts of it that were hidden behind other "
                 "things are missing. Complete it into the whole {piece}, empty, nobody wearing it: fill in only the "
                 "missing parts in the same style, colours and material, and keep every visible part exactly as it is. "
                 "Nothing else in the picture: no body, no skin, no head, no other clothes. Plain flat #808080 grey "
                 "background, soft even studio light, no shadow, no text.")

# the order the layers are made in, inside to outside (the user's, Nov 24): body, clothes, shoes, hands, what the
# hands hold, the bald head, the hair with what is worn on the head, then wings and tails
ORDER = ["clothing", "jewellery", "outerwear", "footwear", "handwear", "hand item", "head", "hair", "attachment"]
HEAD = ("Make the character bald: remove all the hair and everything worn on the head (hats, crowns, headbands, ears, "
        "horns, head wings). Keep the face, the skin, the head shape and everything else exactly as it is, at the "
        "same size and position.")


def the(text):
    """'head with ears' -> 'the head with ears' (a phrase that already starts with a/an/the/her/his stays)."""
    text = text.strip().rstrip(".")
    return text if re.match(r"(?i)(a|an|the|her|his|their|its|both)\b", text) else "the " + text


def slug(text):
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:24] or "part"


def plan_from(answer, name, sources):
    """The plan: 00 the A-pose (the reference: no 3D), 01 the base body, then every piece from the inside to the
    outside (ORDER), with the bald head before the hair, and what is worn on the head made with the hair."""
    keep = answer.get("keep") or "the face, the hair and the outfit"
    props = [p for p in answer.get("props") or [] if isinstance(p, str) and p.strip()]
    leave = f" Leave out {', '.join(props)}." if props else ""
    parts = [
        {"id": "00_apose", "name": "Whole character, A-pose", "kind": "apose", "no3d": True,
         "what": "the reference every layer is cut from and lined up with: alone, standing straight, arms down and out "
                 "at about 45 degrees, legs a little apart, front view" + (", without " + ", ".join(props) if props else ""),
         "prompt": "Show this character alone in an A-pose: standing straight, facing the camera, arms down and away "
                   "from the body at about 45 degrees, palms toward the body, legs slightly apart, feet flat. "
                   f"Keep {keep} exactly as they are.{leave} {LOCK}"},
        {"id": "01_base_body", "name": "Base body", "kind": "base",
         "what": "the same body as the A-pose, 1 to 1 (chest, hips, bottom, legs the same size), with her skin colour, "
                 "smooth (no nipples, no genitals), bald: what every piece is put on",
         "prompt": f"{BASE} {FRAME}"},
    ]
    found = [p for p in answer.get("parts") or [] if isinstance(p, dict) and p.get("show")]
    named = " ".join(str(p.get("name", "")) + " " + str(p.get("show", "")) for p in found).lower()
    def known(text):
        return any(w in named for w in re.findall(r"[a-z]{4,}", text.lower()) if w not in ("with", "giant", "large", "small"))
    for prop in props:  # every held item is a part of its own (the model sometimes lists it only as a prop)
        if not known(prop):
            found.append({"name": prop, "show": f"the {prop}", "category": "hand item"})
            named += " " + prop.lower()
    hair = [p for p in found if str(p.get("category", "")).lower() == "hair"]
    headwear = [p for p in found if str(p.get("category", "")).lower() == "headwear"]
    if not hair and "bald" not in keep.lower():
        hair = [{"name": "hair", "show": "the hair", "category": "hair"}]
    # the hair with what is worn on the head: one part, as it sits on the head (the user's order)
    if hair or headwear:
        shows = [h["show"] for h in hair] + [h["show"] for h in headwear]
        found = [p for p in found if p not in hair and p not in headwear]
        found.append({"name": "Hair" + (" with " + ", ".join(h["name"] for h in headwear) if headwear else ""),
                      "show": " with ".join([shows[0], ", ".join(shows[1:])]) if len(shows) > 1 else shows[0],
                      "category": "hair", "words": shows})
    found.append({"name": "Head, bald", "show": "the bald head with the face and the neck", "category": "head",
                  "kind": "head"})
    rank = lambda p: ORDER.index(p.get("category", "clothing")) if p.get("category") in ORDER else 1
    found.sort(key=rank)  # a stable sort: the model's order stays within a category
    used = {"apose", "base_body"}
    for part in found:
        label = str(part.get("name") or "").strip() or " ".join(part["show"].split()[:3])
        s = slug(label)
        if s in used or s in ("torso", "upper_body", "chest", "center", "centre", "body", "bare_body", "face",
                              "arms", "legs", "hands", "feet"):
            continue
        used.add(s)
        kind = part.get("kind", "piece")
        parts.append({"id": f"{len(parts):02d}_{s}", "name": label[:1].upper() + label[1:], "kind": kind,
                      "category": part.get("category", "clothing"), "what": part["show"],
                      "words": part.get("words") or [part["show"]],
                      "prompt": HEAD if kind == "head" else COMPLETE_TEXT.format(piece=the(part["show"]))})
        if len(parts) >= 14:
            break
    for p in parts:
        p["grey_prompt"] = GREY
    parts[1]["grey_prompt"] = GREY_BASE
    return {"name": name, "character": answer.get("name", ""), "sources": sources, "source": sources[0],
            "locked": LOCKED, "editor": EDITOR,
            "instruction": "Slice this character into layers for 3D, from the inside to the outside: the A-pose (the "
                           "reference), the base body, the clothes, shoes, hands, what the hands hold, the bald head, "
                           "the hair with what is worn on the head, wings and tails. Every piece is cut out of the "
                           "A-pose and completed, so all layers line up.",
            "parts": parts}


def make_plan(folder, cpu=False, note="", model=None):
    from organise_pictures import ask, load, shrink
    from describe_colours import on_grey
    plan_file = os.path.join(folder, "slicer.json")
    old = json.load(open(plan_file)) if os.path.isfile(plan_file) else {}
    if old.get("sources") and len(old["sources"]) > 4:
        print(f"{len(old['sources'])} pictures: Qwen3-VL sees the first 4", flush=True)
    sources = old.get("sources") or sorted(
        os.path.relpath(os.path.join(folder, "sources", f), PROJECT) for f in os.listdir(os.path.join(folder, "sources")))
    model, processor = model or load("Qwen/Qwen3-VL-8B-Instruct", cpu)
    images = [shrink(on_grey(Image.open(os.path.join(PROJECT, s))), 768) for s in sources[:4]]
    question = QUESTION + (f"\nWhat the user says about it (follow it): {note.strip()[:600]}" if note.strip() else "")
    answer, raw = ask(model, processor, images, question, max_tokens=900)
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
    p.add_argument("--note", default="", help="what the user wrote with the pictures")
    a = p.parse_args()
    plan = make_plan(a.folder, a.cpu, a.note)
    for part in plan["parts"]:
        print(f"{part['id']:<22} {part['what'][:100]}")


if __name__ == "__main__":
    main()
