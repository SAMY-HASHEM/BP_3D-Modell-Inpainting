"""Cut parts out of a picture with boxes, for pictures without a colour-ID render.

Each part is a box in the picture plus a rule for which pixels inside the box belong to it:
    notskin   everything of the figure except skin (clothes, boots, hair over a face, ...)
    dark      only dark pixels (black leather, dark fabric)
    hair      dark pixels that are not very colourful (dark hair, without the reddish shadows on skin)
    all       everything of the figure inside the box
Pieces that don't touch become separate parts (e.g. two boots in one box: boot-1 and boot-2).

Usage:
    source activate.sh
    python scripts/mask/cut_regions.py pictures/obsession/clean.png \\
        --part hair 880 380 1600 1060 hair \\
        --part boot 980 2640 1520 3680 notskin

Writes <picture folder>/parts/<name>.png (RGBA cut-outs for generate.py and insert_part.py)
and overview.jpg to check them.
"""
import _paths  # noqa: F401  (the scripts of the other folders)
import argparse
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw

from cut_parts import fill_holes
from project_texture import object_masks

RULES = ("notskin", "dark", "hair", "all")


def skin(picture):
    """Skin-coloured pixels: orange-ish hue, medium saturation, not dark (red fabric is more saturated)."""
    hsv = cv2.cvtColor(picture, cv2.COLOR_RGB2HSV).astype(int)
    hue, saturation, value = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return ((hue <= 25) | (hue >= 170)) & (saturation >= 40) & (saturation <= 170) & (value >= 90)


def main():
    parser = argparse.ArgumentParser(description="Cut parts out of a picture with boxes")
    parser.add_argument("picture")
    parser.add_argument("--part", nargs=6, action="append", required=True,
                        metavar=("NAME", "X0", "Y0", "X1", "Y1", "RULE"),
                        help=f"a part: name, box in picture pixels, rule ({', '.join(RULES)}); repeat for more")
    parser.add_argument("--out", help="folder for the cut-outs (default: <picture folder>/parts)")
    parser.add_argument("--min-size", type=float, default=0.2,
                        help="drop pieces smaller than this fraction of the part's biggest piece (default 0.2)")
    args = parser.parse_args()
    out = args.out or os.path.join(os.path.dirname(args.picture), "parts")
    os.makedirs(out, exist_ok=True)

    image = Image.open(args.picture).convert("RGB")
    picture = np.asarray(image)
    print("Cutting the figure out of the picture...")
    figure = object_masks([image])[0] > 0.5
    is_skin = skin(picture)
    hsv = cv2.cvtColor(picture, cv2.COLOR_RGB2HSV)
    is_dark = hsv[..., 2] < 90
    is_hair = is_dark & (hsv[..., 1] < 125)  # shadows on skin are dark too, but more saturated

    overview = Image.fromarray((picture * 0.5).astype(np.uint8))
    for name, *box, rule in args.part:
        if rule not in RULES:
            parser.error(f"unknown rule {rule!r} for {name} (use {', '.join(RULES)})")
        x0, y0, x1, y1 = map(int, box)
        mask = np.zeros_like(figure)
        mask[y0:y1, x0:x1] = figure[y0:y1, x0:x1]
        if rule == "notskin":
            mask &= ~is_skin
        elif rule == "dark":
            mask &= is_dark
        elif rule == "hair":
            mask &= is_hair
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        if count < 2:
            print(f"    {name}: nothing found in the box, skipped")
            continue
        biggest = stats[1:, cv2.CC_STAT_AREA].max()
        pieces = [j for j in range(1, count) if stats[j, cv2.CC_STAT_AREA] >= args.min_size * biggest]
        pieces.sort(key=lambda j: stats[j, cv2.CC_STAT_LEFT])
        for n, j in enumerate(pieces, start=1):
            piece_name = name if len(pieces) == 1 else f"{name}-{n}"
            piece = fill_holes(labels == j)
            piece = cv2.dilate(piece.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & figure
            ys, xs = np.nonzero(piece)
            pad = 16
            px0, px1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, picture.shape[1])
            py0, py1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, picture.shape[0])
            alpha = (cv2.GaussianBlur(piece.astype(np.float32), (0, 0), 1.0) * 255).astype(np.uint8)
            cutout = np.dstack([picture, alpha])[py0:py1, px0:px1]
            Image.fromarray(cutout, "RGBA").save(os.path.join(out, f"{piece_name}.png"))
            print(f"    {piece_name:10s} {px1 - px0:4d} x {py1 - py0:4d} px")

            shown = np.asarray(overview).copy()
            shown[piece] = picture[piece]
            overview = Image.fromarray(shown)
            draw = ImageDraw.Draw(overview)
            draw.rectangle([px0, py0, px1, py1], outline=(255, 255, 0), width=4)
            draw.text((px0 + 6, py0 + 6), piece_name, fill=(255, 255, 0))
    overview.save(os.path.join(out, "overview.jpg"), quality=90)
    print(f"Saved the parts and overview.jpg to {out}")


if __name__ == "__main__":
    main()
