"""The picture of one rig-head example (rig_data.py) or prediction (rig_from_latent.py): the latent's voxels coloured by
the bone with the most weight, and the 22 joints with their bones, from the front and from the character's left.

    python scripts/rigging/rig_picture.py datasets/rig_latents/02061.npz [more.npz ...]     # writes 02061.jpg next to it
"""
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from rig_data import BONES, GRID, PARENT  # noqa: E402

PALETTE = [(230, 90, 80), (90, 170, 240), (110, 200, 120), (240, 190, 60), (190, 120, 230), (90, 210, 200),
           (240, 140, 60), (160, 160, 170), (230, 110, 170), (150, 200, 80), (120, 110, 240), (200, 170, 130),
           (70, 140, 110), (250, 240, 120), (255, 120, 120), (120, 220, 255), (180, 255, 150), (255, 200, 120),
           (220, 160, 255), (140, 255, 230), (255, 170, 90), (200, 200, 210)]


def picture(path, size=600):
    d = np.load(path)
    centres = (d["coords"].astype(float) + 0.5) / GRID - 0.5  # TRELLIS.2's cube: x right, -y front, z up
    bone = d["skin"].astype(float).argmax(1)
    joints = d["joints"].astype(float)
    sheet = Image.new("RGB", (2 * size, size + 30), (30, 30, 34))
    draw = ImageDraw.Draw(sheet)
    # front: looking along +y (from -y) -> screen x = cube x; side: from the character's left (+x) -> screen x = -y
    for k, (across, depth) in enumerate(((0, 1), (1, 0))):
        sign = 1 if k == 0 else -1
        spot = lambda p: (k * size + size / 2 + sign * p[..., across] * size * 0.95, size / 2 - p[..., 2] * size * 0.95)
        near = -centres[:, depth] if k == 0 else centres[:, depth]
        for i in np.argsort(-near):  # far ones first
            x, y = spot(centres[i])
            r = size / GRID / 2
            draw.rectangle((x - r, y - r, x + r, y + r), fill=PALETTE[bone[i] % len(PALETTE)])
        for j, p in enumerate(PARENT):
            if p >= 0:
                draw.line([spot(joints[p]), spot(joints[j])], fill=(255, 255, 255), width=3)
        if "fingers" in d:  # each finger: hand -> 1 -> 2 -> 3, in yellow
            fingers, has = d["fingers"].astype(float), d["has_fingers"]
            for f in range(10):
                chain = [joints[9 if f < 5 else 13]] + [fingers[3 * f + k] for k in range(3) if has[3 * f + k]]
                for a, b in zip(chain, chain[1:]):
                    draw.line([spot(a), spot(b)], fill=(255, 220, 60), width=2)
        for j in range(len(BONES)):
            x, y = spot(joints[j])
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(0, 0, 0), outline=(255, 255, 255))
        draw.text((k * size + 10, size + 8), ("front" if k == 0 else "from its left") + f"  {path.rsplit('/', 1)[-1]}",
                  fill=(220, 220, 220))
    out = path.rsplit(".", 1)[0] + ".jpg"
    sheet.save(out, quality=90)
    return out


if __name__ == "__main__":
    for path in sys.argv[1:]:
        print(picture(path))
