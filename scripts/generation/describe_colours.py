"""A short description of an object's colours, part by part, from its pictures, by a local vision-language model
(Qwen3-VL-8B): the text MV-Adapter paints by (texture_mvadapter.py --text). Without it MV-Adapter guessed what it
could not see (the motorcycle girl's bike came out green, Nov 23); a hand-written description fixed most of it.

    python scripts/generation/describe_colours.py front.png side.png back.png [--out colours.txt]

Prints the description (at most ~60 words: SDXL reads 77 tokens) and writes it to --out.
--parts: instead the parts and their materials -> detail_normals.py's --parts (one per line in --out).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths  # noqa: F401,E402

os.environ.setdefault("HF_HUB_CACHE", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "local_ai_models"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from PIL import Image  # noqa: E402

from organise_pictures import ask, load, shrink  # noqa: E402

QUESTION = """These pictures show one object (a character, a vehicle, a sculpture ...) from different sides.
Describe its COLOURS and MATERIALS part by part, for an artist who paints a 3D model of it: name each visible part
and its colour (and material if clear: metal, leather, cloth, skin, stone ...). Be exact about colours that could be
confused (e.g. an orange helmet with a pale green visor). One sentence, at most 55 words, no style words, no
background. Answer only with JSON: {"description": "..."}"""


PARTS_QUESTION = """These pictures show one object from different sides. List its main visible PARTS (at most 5,
each a word a part finder can look for: "helmet", "shirt", "face", "crown", "boots" ...) and each one's material, one
of: skin, cotton, denim, wool, satin, leather, metal, plastic, stone, rubber, hair. Answer only with JSON:
{"parts": [{"word": "...", "material": "..."}]}"""

# material -> a detail map (local_ai_models/detail_maps or noise), its strength, its tile (share of the model)
DETAIL = {"skin": ("noise", 0.35, 0.012), "cotton": ("cotton_jersey", 0.6, 0.03), "denim": ("denim_fabric", 0.6, 0.03),
          "wool": ("caban", 0.7, 0.03), "satin": ("crepe_satin", 0.4, 0.03), "leather": ("brown_leather", 0.6, 0.03),
          "metal": ("noise", 0.3, 0.03), "plastic": ("noise", 0.15, 0.05), "stone": ("noise", 0.4, 0.015),
          "rubber": ("rubberized_track", 0.5, 0.03)}


def parts_for_detail(pictures, model_name="Qwen/Qwen3-VL-8B-Instruct", cpu=False):
    """detail_normals.py's --parts from the pictures: Qwen3-VL names the parts and their materials, DETAIL gives
    each material's map (hair: none)."""
    model, processor = load(model_name, cpu)
    answer, raw = ask(model, processor, [shrink(on_grey(Image.open(p)), 768) for p in pictures], PARTS_QUESTION,
                      max_tokens=300)
    specs = []
    for part in answer.get("parts", [])[:5]:
        material = str(part.get("material", "")).lower()
        if material in DETAIL and part.get("word"):
            name, strength, tile = DETAIL[material]
            specs.append(f"{part['word']}={name}:{strength}:{tile}")
    return specs


def on_grey(im):
    if im.mode == "RGBA":  # cut-outs: on grey, as MV-Adapter sees them
        bg = Image.new("RGB", im.size, (127, 127, 127))
        bg.paste(im, mask=im.getchannel("A"))
        return bg
    return im.convert("RGB")


def describe(pictures, model_name="Qwen/Qwen3-VL-8B-Instruct", cpu=False):
    model, processor = load(model_name, cpu)
    images = [shrink(on_grey(Image.open(path)), 768) for path in pictures]
    answer, raw = ask(model, processor, images, QUESTION, max_tokens=300)
    text = (answer.get("description") or raw).strip().replace("\n", " ")
    words = text.split()
    return " ".join(words[:60])


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("pictures", nargs="+")
    p.add_argument("--out")
    p.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--parts", action="store_true", help="instead: the parts and their detail maps for detail_normals.py")
    a = p.parse_args()
    if a.parts:
        specs = parts_for_detail(a.pictures, a.model, a.cpu)
        print(" ".join(f'"{x}"' for x in specs))
        if a.out:
            with open(a.out, "w") as f:
                f.write("\n".join(specs) + "\n")
        return
    text = describe(a.pictures, a.model, a.cpu)
    print(text)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
