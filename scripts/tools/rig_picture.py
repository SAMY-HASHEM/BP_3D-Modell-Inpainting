"""The picture of a named rig (rig_parts.py): the parts in colours and the bones on them, from the front and from the
character's right side.    python scripts/tools/rig_picture.py outputs/x/rigged     (reads rigged_draw.npz, writes rigged.jpg)"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

stem = sys.argv[1]
d = np.load(stem + "_draw.npz")
points, low, high, front, left = d["points"], d["low"], d["high"], d["front"], d["left"]
S = 900
centre, scale = (low + high) / 2, 0.86 * S / float((high - low).max())
palette = [(230, 90, 80), (90, 170, 240), (110, 200, 120), (240, 190, 60), (190, 120, 230), (90, 210, 200), (240, 140, 60),
           (160, 160, 170), (230, 110, 170), (150, 200, 80), (120, 110, 240), (200, 170, 130), (70, 140, 110), (250, 240, 120)]
colour = {str(part): palette[k % len(palette)] for k, part in enumerate(d["parts"])}
side = left / np.linalg.norm(left)
sheet = Image.new("RGB", (2 * S, S + 40), (30, 30, 34))
draw = ImageDraw.Draw(sheet)
for k, (across, depth) in enumerate(((side, front), (front, -side))):  # its left is on the right of a front view; depth: towards the viewer
    spot = lambda q: (k * S + S / 2 + float((q - centre) @ across) * scale, S / 2 - float(q[2] - centre[2]) * scale)
    order = np.argsort(points @ depth)  # the far points first, the near ones on top
    for i in order[::max(1, len(order) // 80000)]:
        x, y = spot(points[i])
        fill = colour.get(str(d["part"][i]), (120, 120, 130)) if len(d["part"]) else (120, 125, 135)
        draw.rectangle((x - 1, y - 1, x + 1, y + 1), fill=fill)
    for a, e in d["lines"]:
        draw.line([spot(a), spot(e)], fill=(255, 255, 255), width=2)
    for joint, name in zip(d["joints"], d["joint_names"]):
        x, y = spot(joint)
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(20, 20, 20), outline=(255, 255, 255))
        draw.text((x + 6, y - 5), str(name), fill=(255, 255, 255))
    draw.text((k * S + 10, 8), "from the front" if k == 0 else "from its right side", fill=(255, 255, 255))
x = 10
for part, c in colour.items():  # the legend
    draw.rectangle((x, S + 12, x + 14, S + 26), fill=c)
    draw.text((x + 20, S + 13), part, fill=(235, 235, 235))
    x += 40 + 7 * len(part)
sheet.save(stem + ".jpg", quality=92)
os.remove(stem + "_draw.npz")
