"""A part from words gets a picture of its own: what the picture AI painted, cut out of the edit's AFTER picture.

make_edit.py --prompt paints the words onto the model's front picture; what differs from the BEFORE picture is the 2D
mask. A part made that way had only that one front and was built from noise inside the model: cora's tiara came out
as a thin ring of wires (Nov 19, part 27). A part with a picture of its own goes the way that worked for the toaster's
book instead: its other sides from the picture AI, its own 3D model, placed in 3D, only the details step on it, its
colours from its own model. This writes that picture (the part on grey, square, with a margin) and a data sheet as
data_sheet.py writes one, so the flow's part nodes take it as they take an uploaded picture.

    python scripts/inpainting/cut_part.py outputs/x/work/03_edit/<id> --out outputs/x/work/03a_cut/<id> [--what "a golden tiara"]
"""
import argparse
import json
import os

import numpy as np
from PIL import Image
from scipy import ndimage

GREY = (190, 190, 192)


def main():
    parser = argparse.ArgumentParser(description="the painted part cut out of the edit's AFTER picture")
    parser.add_argument("edit", help="the folder make_edit.py wrote (after.png, mask.png, edit.json)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--what", default="", help="what the part is (the data sheet's what); default the edit's prompt")
    parser.add_argument("--margin", type=float, default=0.15, help="room around the part, a share of its size")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    after = np.asarray(Image.open(os.path.join(args.edit, "after.png")).convert("RGB"))
    mask = np.asarray(Image.open(os.path.join(args.edit, "mask.png")).convert("L")) > 127
    labels, count = ndimage.label(mask)
    if not count:
        raise SystemExit("the edit's mask is empty: nothing was painted")
    sizes = ndimage.sum(mask, labels, range(1, count + 1))
    mask = np.isin(labels, 1 + np.nonzero(sizes >= 0.05 * sizes.max())[0])  # specks the picture AI changed by the way go
    ys, xs = np.nonzero(mask)
    side = max(np.ptp(ys), np.ptp(xs)) * (1 + 2 * args.margin)
    middle = ((ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2)
    part = np.where(mask[..., None], after, np.array(GREY, np.uint8))
    canvas = Image.new("RGB", (int(side), int(side)), GREY)
    top, left = int(middle[0] - side / 2), int(middle[1] - side / 2)
    canvas.paste(Image.fromarray(part).crop((left, top, left + int(side), top + int(side))), (0, 0))
    picture = os.path.join(args.out, "part.png")
    canvas.resize((1024, 1024), Image.LANCZOS).save(picture)
    what = " ".join(args.what.split()) or json.load(open(os.path.join(args.edit, "edit.json"))).get("prompt") or "the new part"
    json.dump({"what": what, "views": [{"name": "part", "path": os.path.abspath(picture), "side": "front", "made": False}],
               "lists": {"base": ["part"], "sides": ["part"], "why": ["the part the picture AI painted, cut out of the AFTER picture"]}},
              open(os.path.join(args.out, "sheet.json"), "w"), indent=1)
    print(f"the part: {len(ys):,} pixels of the AFTER picture, {picture}", flush=True)


if __name__ == "__main__":
    main()
