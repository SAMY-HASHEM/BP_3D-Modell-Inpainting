"""Turn a picture and its part cut-outs into OmniPart's input: an RGBA picture and a part-id mask.

OmniPart (github.com/HKU-MMLab/OmniPart) generates a 3D model made of parts. It needs every pixel of
the object to carry a part number (0 = background), saved as an .exr file. This script takes the
cut-outs made by cut_parts.py or cut_regions.py, finds each one in the picture, and gives it its own
number; everything else of the object becomes part 1 ("body").

Usage:
    source activate.sh
    python scripts/mask/make_part_mask.py pictures/yeonhee/parts-front.png pictures/yeonhee/parts-auto

Writes next to the picture: <name>_rgba.png, <name>_mask.exr and <name>_mask_preview.png.
OmniPart shrinks the mask to 37 x 37 cells, so parts smaller than about 1/37 of the object vanish.
"""
import _paths  # noqa: F401  (the scripts of the other folders)
import argparse
import glob
import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2
import numpy as np
from PIL import Image

from insert_part import locate
from project_texture import object_masks


def main():
    parser = argparse.ArgumentParser(description="Make OmniPart's RGBA picture and part mask")
    parser.add_argument("picture", help="the full picture")
    parser.add_argument("parts_dir", help="folder with the RGBA part cut-outs of this picture")
    args = parser.parse_args()

    image = Image.open(args.picture).convert("RGB")
    picture = np.asarray(image)
    height, width = picture.shape[:2]
    print("Cutting the object out of the picture...")
    alpha = object_masks([image])[0]
    figure = alpha > 0.5

    mask = np.where(figure, 1, 0).astype(np.int32)  # 1 = everything not covered by a part ("body")
    names = ["body"]
    for path in sorted(glob.glob(os.path.join(args.parts_dir, "*.png"))):
        cutout = Image.open(path).convert("RGBA")
        x0, y0 = locate(cutout, image)
        part = np.asarray(cutout)[..., 3] > 127
        region = mask[y0:y0 + part.shape[0], x0:x0 + part.shape[1]]
        region[part & figure[y0:y0 + part.shape[0], x0:x0 + part.shape[1]]] = len(names) + 1
        names.append(os.path.splitext(os.path.basename(path))[0])

    base = os.path.splitext(args.picture)[0]
    rgba = np.dstack([picture, (alpha * 255).astype(np.uint8)])
    Image.fromarray(rgba, "RGBA").save(base + "_rgba.png")
    cv2.imwrite(base + "_mask.exr", np.repeat(mask[..., None], 3, axis=2).astype(np.float32))

    colors = np.random.default_rng(0).integers(60, 255, size=(len(names) + 1, 3)).astype(np.uint8)
    colors[0] = 0
    Image.fromarray(colors[mask]).save(base + "_mask_preview.png")
    for i, name in enumerate(names, start=1):
        share = (mask == i).sum() / figure.sum()
        warning = "  (very small: may vanish in OmniPart's 37 x 37 grid)" if share < 0.01 else ""
        print(f"    part {i:2d}: {name:10s} {share:5.1%} of the object{warning}")
    print(f"Saved {base}_rgba.png, {base}_mask.exr and {base}_mask_preview.png")


if __name__ == "__main__":
    main()
