"""Cut a picture into parts, using a colour-ID render of the same pose (every part has its own colour).

Each part becomes an RGBA cut-out of the textured picture, ready for generate.py and insert_part.py.
Only clearly coloured parts are cut (red, orange, yellow, green, teal, blue, purple, pink); white,
grey, skin, hair and dark areas stay with the body. Parts of the same colour that don't touch are separate
parts (e.g. left and right boot).

Usage:
    source activate.sh
    python scripts/cut_parts.py pictures/yeonhee/parts-colors.png pictures/yeonhee/parts-front.png
    python scripts/cut_parts.py colors.png picture.png --out pictures/yeonhee/parts --min-size 0.01
    # no colour render: a part that already has its own colour (here turquoise hair) in the picture itself
    python scripts/cut_parts.py pictures/head/head.png pictures/head/head.png --only teal --min-saturation 80 --close 25

Writes <out>/<colour>-<n>.png (numbered left to right) and <out>/overview.jpg to check the parts.

Steps:
    1. cut the figure out of both pictures (RMBG-2.0)
    2. align the colour render to the picture (they may be shifted, e.g. two halves of one sheet)
    3. find the parts: pixels of one colour, split into pieces that don't touch
    4. fine-align each part on the edges of the picture (the two renders differ by a few pixels)
    5. save each part as a cut-out of the picture, with a transparent background
"""
import argparse
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw

from project_texture import object_masks

# OpenCV hue (0..180) of the colours used in colour-ID renders
HUES = {"red": 0, "orange": 15, "yellow": 30, "green": 60, "teal": 85, "blue": 110, "purple": 135, "pink": 160}
MAX_HUE_DISTANCE = 10  # pixels further than this from every colour above belong to no part
# ID colours are much more saturated than skin, hair or leather; pastel purple is the exception
MIN_SATURATION = {"purple": 90}
DEFAULT_MIN_SATURATION = 160
SEARCH = 24  # fine alignment: how many pixels a part may be shifted in each direction


def align(moving, fixed):
    """Shift (dx, dy) that moves the mask `moving` onto `fixed` (phase correlation)."""
    height, width = max(moving.shape[0], fixed.shape[0]), max(moving.shape[1], fixed.shape[1])
    pad = lambda m: np.pad(m, ((0, height - m.shape[0]), (0, width - m.shape[1]))).astype(np.float32)
    blur = lambda m: cv2.GaussianBlur(pad(m), (0, 0), 4)
    (dx, dy), _ = cv2.phaseCorrelate(blur(moving), blur(fixed))
    return dx, dy


def shift_mask(mask, dx, dy, shape):
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(mask.astype(np.uint8), matrix, (shape[1], shape[0]), flags=cv2.INTER_NEAREST) > 0


def fine_align(mask, edges):
    """Shift the part's outline to where the picture has the most edges (within +-SEARCH pixels)."""
    outline = mask & ~cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    ys, xs = np.nonzero(outline)
    height, width = edges.shape
    best, best_shift = -1, (0, 0)
    for dy in range(-SEARCH, SEARCH + 1, 2):
        for dx in range(-SEARCH, SEARCH + 1, 2):
            y, x = ys + dy, xs + dx
            ok = (y >= 0) & (y < height) & (x >= 0) & (x < width)
            score = edges[y[ok], x[ok]].sum()
            if score > best:
                best, best_shift = score, (dx, dy)
    return best_shift


def fill_holes(mask):
    """Fill holes that lie completely inside a part (e.g. bright highlights that lost their colour)."""
    inverse = (~mask).astype(np.uint8)
    count, labels = cv2.connectedComponents(inverse)
    touching = set(np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])))
    holes = ~np.isin(labels, list(touching)) & ~mask
    return mask | holes


def find_parts(colors, min_area, only=None, min_saturation=None, close=7):
    """Masks of all clearly coloured parts in the colour render, with their colour names."""
    hsv = cv2.cvtColor(colors, cv2.COLOR_RGB2HSV).astype(int)
    hue, saturation, value = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    names = list(HUES)
    distance = np.stack([np.minimum(np.abs(hue - h), 180 - np.abs(hue - h)) for h in HUES.values()])
    nearest = distance.argmin(0)
    parts = []
    for i, name in enumerate(names):
        if only and name not in only:
            continue
        limit = min_saturation if min_saturation is not None else MIN_SATURATION.get(name, DEFAULT_MIN_SATURATION)
        colored = (saturation >= limit) & (value > 50)
        mask = colored & (nearest == i) & (distance[i] <= MAX_HUE_DISTANCE)
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((close, close), np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        pieces = [j for j in range(1, count) if stats[j, cv2.CC_STAT_AREA] >= min_area]
        pieces.sort(key=lambda j: stats[j, cv2.CC_STAT_LEFT] + stats[j, cv2.CC_STAT_WIDTH] / 2)
        for n, j in enumerate(pieces, start=1):
            parts.append((f"{name}-{n}", fill_holes(labels == j)))
    return parts


def main():
    parser = argparse.ArgumentParser(description="Cut a picture into parts using a colour-ID render")
    parser.add_argument("colors", help="colour-ID render: every part in its own clear colour")
    parser.add_argument("picture", help="the textured picture in the same pose")
    parser.add_argument("--out", help="folder for the cut-outs (default: <picture folder>/parts)")
    parser.add_argument("--min-size", type=float, default=0.005,
                        help="smallest part, as a fraction of the figure's area (default 0.005)")
    parser.add_argument("--only", nargs="+", choices=list(HUES), help="cut only parts of these colours")
    parser.add_argument("--close", type=int, default=7,
                        help="close gaps up to this many pixels inside a part (default 7; e.g. 25 for hair)")
    parser.add_argument("--min-saturation", type=int,
                        help=f"0..255, how colourful a pixel must be to count (default {DEFAULT_MIN_SATURATION}; "
                             f"lower it for parts with a natural, less saturated colour)")
    args = parser.parse_args()
    out = args.out or os.path.join(os.path.dirname(args.picture), "parts")
    os.makedirs(out, exist_ok=True)

    colors = np.asarray(Image.open(args.colors).convert("RGB"))
    picture_image = Image.open(args.picture).convert("RGB")
    picture = np.asarray(picture_image)

    print("1. Cutting the figure out of both pictures...")
    color_mask, picture_mask = (m > 0.5 for m in object_masks([Image.fromarray(colors), picture_image]))

    dx, dy = align(color_mask, picture_mask)
    print(f"2. The colour render is shifted by ({dx:+.0f}, {dy:+.0f}) pixels against the picture")

    print("3. Finding the parts...")
    parts = find_parts(colors, args.min_size * picture_mask.sum(), args.only, args.min_saturation, args.close)
    gray = cv2.GaussianBlur(cv2.cvtColor(picture, cv2.COLOR_RGB2GRAY), (0, 0), 1.5)
    edges = cv2.dilate(cv2.Canny(gray, 30, 90), np.ones((3, 3), np.uint8)) > 0

    overview = Image.fromarray((picture * 0.5).astype(np.uint8))
    draw = ImageDraw.Draw(overview)
    for name, mask in parts:
        mask = shift_mask(mask, dx, dy, picture.shape)
        fx, fy = fine_align(mask, edges)
        mask = shift_mask(mask, fx, fy, picture.shape)
        mask = cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & picture_mask
        ys, xs = np.nonzero(mask)
        pad = 16
        x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, picture.shape[1])
        y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, picture.shape[0])
        alpha = (cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 1.0) * 255).astype(np.uint8)
        Image.fromarray(np.dstack([picture, alpha])[y0:y1, x0:x1], "RGBA").save(os.path.join(out, f"{name}.png"))
        print(f"    {name:9s} {x1 - x0:4d} x {y1 - y0:4d} px  (fine shift {fx:+d}, {fy:+d})")

        # overview: the part in full colour with its outline and name
        part = np.asarray(overview).copy()
        part[mask] = picture[mask]
        overview = Image.fromarray(part)
        draw = ImageDraw.Draw(overview)
        draw.rectangle([x0, y0, x1, y1], outline=(255, 255, 0), width=3)
        draw.text((x0 + 4, y0 + 4), name, fill=(255, 255, 0))
    overview.convert("RGB").save(os.path.join(out, "overview.jpg"), quality=90)
    print(f"Saved {len(parts)} parts and overview.jpg to {out}")


if __name__ == "__main__":
    main()
