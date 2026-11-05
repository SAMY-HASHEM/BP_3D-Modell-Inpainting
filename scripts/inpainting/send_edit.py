"""Sends what make_edit.py made to the uni server, with the settings that worked best (Nov 5, crown and helmet):
new noise in the area, soft edge 2, colour guidance 3, every side with its own before, after and mask.

    python scripts/inpainting/send_edit.py outputs/edits/robot_crown model.glb --seeds 1 2     # the shape
    python scripts/inpainting/send_edit.py outputs/edits/robot_crown new.glb --colours         # then the colours

--colours keeps the shape of new.glb (the result of the shape step) and regenerates only the colours in the area, on
the same mesh: its before pictures are rendered first, in the same frame as the edit's.
"""
import _paths  # noqa: F401  (the scripts of the other folders)

import argparse
import json
import os
import subprocess
import sys

from PIL import Image

from make_edit import BLENDER, PROJECT, on_grey

SETTINGS = "inversion,start=random,soft=2,texture_guidance=3"


def main():
    parser = argparse.ArgumentParser(description="Send a make_edit.py edit to the uni server")
    parser.add_argument("edit", help="the folder make_edit.py wrote")
    parser.add_argument("model")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1])
    parser.add_argument("--colours", action="store_true", help="model is the shape step's result: only its colours")
    parser.add_argument("--code", help="run it with the newest code of this branch")
    args = parser.parse_args()
    edit = json.load(open(os.path.join(args.edit, "edit.json")))
    afters = [os.path.join(args.edit, "after.png")] + edit["more_edit"][1::4]
    befores = [os.path.join(args.edit, "before.png")] + edit["more_edit"][2::4]
    masks = [os.path.join(args.edit, "mask.png")] + edit["more_edit"][3::4]
    variant = f"{SETTINGS},room={edit['room']}"
    grow = "0"
    if args.colours:
        folder = os.path.join(args.edit, "colours")
        os.makedirs(folder, exist_ok=True)
        subprocess.run([BLENDER, "-b", "--python", os.path.join(PROJECT, "scripts/tools/render_views.py"), "--",
                        os.path.abspath(args.model), os.path.abspath(folder), "--frame",
                        os.path.abspath(os.path.join(args.edit, "views.json"))], check=True, stdout=subprocess.DEVNULL)
        befores = []
        for k in range(len(afters)):
            view = Image.open(os.path.join(folder, f"view_{(edit['side'] + k) % 4}.png")).convert("RGBA")
            befores.append(os.path.join(folder, f"before_{k}.png"))
            on_grey(view).save(befores[-1])
        variant = f"{SETTINGS},keep_shape=1"
    command = [sys.executable, os.path.join(PROJECT, "scripts/server/cloud.py"), "inpaint", args.model, afters[0],
               "--area", os.path.join(args.edit, "area.npy"), "--source", befores[0], "--picture-mask", masks[0],
               "--grow", grow, "--steps", "25", "--variant", variant, "--seeds", *map(str, args.seeds)]
    for side in zip(afters[1:], befores[1:], masks[1:]):
        command += ["--more-edit", *side]
    if args.code:
        command += ["--code", args.code]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
