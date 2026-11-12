"""The AFTER picture of an edit from a text prompt, for the edit by hand of the web page (flow_page.py).

make_edit.py made BEFORE (the model from one side, on grey). Here the local picture AI (Qwen-Image-Edit-2511 through
ComfyUI, as picture_from_prompt.py) paints the prompt into the painted area of that picture; outside the area the
picture stays the exact BEFORE, so only the area changes. make_edit.py --after then takes it instead of a collage.

    source activate.sh
    python scripts/inpainting/prompt_picture.py before.png area.png out.png --prompt "a golden crown with pink gems" \\
        [--reference crown.png] [--seed 1]

area.png: white (or painted) = where the change may be, the rest black or see-through.
"""
import _paths  # noqa: F401  (the scripts of the other folders)

import argparse

import cv2
import numpy as np
from PIL import Image

from picture_from_prompt import FRAME, comfy_edit


def main():
    parser = argparse.ArgumentParser(description="Paint a prompt into the painted area of a picture (local AI)")
    parser.add_argument("before")
    parser.add_argument("area", help="picture of the area: white or painted = may change")
    parser.add_argument("out")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--reference", help="optional picture of the thing to add")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    before = np.asarray(Image.open(args.before).convert("RGB"))
    size = before.shape[0]
    painted = Image.open(args.area)
    painted = painted.getchannel("A") if painted.mode == "RGBA" else painted.convert("L")  # the page's layers are see-through
    area = np.asarray(painted.resize((size, size), Image.NEAREST)) > 127
    if not area.any():
        raise SystemExit("nothing is painted: paint the area where the change should be first")
    ys, xs = np.nonzero(area)
    pad = int(0.02 * size)
    x0, y0, x1, y1 = max(0, xs.min() - pad), max(0, ys.min() - pad), min(size - 1, xs.max() + pad), min(size - 1, ys.max() + pad)
    marked = before.copy()
    cv2.rectangle(marked, (int(x0), int(y0)), (int(x1), int(y1)), FRAME, max(3, size // 200))
    reference = Image.open(args.reference) if args.reference else None
    report = lambda text: print(text, flush=True)
    edited = np.asarray(comfy_edit(Image.fromarray(marked), args.prompt, args.seed, reference, report).resize((size, size), Image.LANCZOS))

    # only the painted area changes (soft edge): the rest stays the exact BEFORE
    grow = max(4, size // 100)
    region = cv2.dilate(area.astype(np.float32), np.ones((2 * grow + 1, 2 * grow + 1), np.uint8))
    region = cv2.GaussianBlur(region, (0, 0), grow / 2)[..., None]
    Image.fromarray((edited * region + before * (1 - region)).astype(np.uint8)).save(args.out)
    print(f"Result: {args.out}", flush=True)


if __name__ == "__main__":
    main()
