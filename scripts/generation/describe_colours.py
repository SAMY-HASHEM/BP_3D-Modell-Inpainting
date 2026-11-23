"""A short description of an object's colours, part by part, from its pictures, by a local vision-language model
(Qwen3-VL-8B): the text MV-Adapter paints by (texture_mvadapter.py --text). Without it MV-Adapter guessed what it
could not see (the motorcycle girl's bike came out green, Nov 23); a hand-written description fixed most of it.

    python scripts/generation/describe_colours.py front.png side.png back.png [--out colours.txt]

Prints the description (at most ~60 words: SDXL reads 77 tokens) and writes it to --out.
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


def describe(pictures, model_name="Qwen/Qwen3-VL-8B-Instruct", cpu=False):
    model, processor = load(model_name, cpu)
    images = []
    for path in pictures:
        im = Image.open(path)
        if im.mode == "RGBA":  # cut-outs: on grey, as MV-Adapter sees them
            bg = Image.new("RGB", im.size, (127, 127, 127))
            bg.paste(im, mask=im.getchannel("A"))
            im = bg
        images.append(shrink(im.convert("RGB"), 768))
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
    a = p.parse_args()
    text = describe(a.pictures, a.model, a.cpu)
    print(text)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
