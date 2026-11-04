"""A picture with several views of one object side by side (a turnaround sheet) split into single views.

TRELLIS.2 reads one picture as one object: four helmets in a row gave a mix of all four (Nov 4). Here the
background AI (the same one TRELLIS.2 uses) finds the object pixels; every separate blob of them is one view.
Measured, no AI that names anything: a blob counts if it has at least a quarter of the biggest one's pixels
(small logos and text drop out).

    views = split(pipeline, "pictures/helmet/sheet.jpg", out_dir)   # [] = only one view: use the picture as it is
"""
import os

import numpy as np
from PIL import Image


def object_mask(pipeline, image):
    if pipeline.low_vram:
        pipeline.rembg_model.to(pipeline.device)
    rgba = np.asarray(pipeline.rembg_model(image))
    if pipeline.low_vram:
        pipeline.rembg_model.cpu()
    return rgba[..., 3] > 128


def boxes(mask, shrink=3, share=0.25):
    """The views' boxes: separate blobs of object pixels. The mask is first shrunk a little, so that a thin cable or
    antenna touching the next view doesn't join two views; a blob counts if it has at least `share` of the biggest
    one's pixels (logos, text and loose small bits drop out)."""
    from scipy import ndimage
    core = ndimage.binary_erosion(mask, iterations=shrink)
    labels, count = ndimage.label(core)
    if count < 2:
        return []
    sizes = ndimage.sum(core, labels, range(1, count + 1))
    keep = [k for k in np.argsort(-sizes) if sizes[k] >= share * sizes.max()]  # the biggest first
    found = []
    for k, (rows, cols) in enumerate(ndimage.find_objects(labels)):
        if k in keep:
            found.append((keep.index(k), (cols.start - shrink, rows.start - shrink, cols.stop + shrink, rows.stop + shrink)))
    return [box for _, box in sorted(found)]


def split(pipeline, path, out_dir, name="inpaint_view"):
    """The single views of a sheet, saved as <name><k>.png (biggest first); [] when it shows one view."""
    image = Image.open(path).convert("RGB")
    scale = min(1.0, 1024 / max(image.size))
    small = image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS)
    found = boxes(object_mask(pipeline, small))
    if len(found) < 2:
        return []
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for k, (x0, y0, x1, y1) in enumerate(found):
        pad = 0.03 * max(x1 - x0, y1 - y0)
        box = [max(0, round((x0 - pad) / scale)), max(0, round((y0 - pad) / scale)),
               min(image.width, round((x1 + pad) / scale)), min(image.height, round((y1 + pad) / scale))]
        paths.append(os.path.join(out_dir, f"{name}{k + 1}.png"))
        image.crop(box).save(paths[-1])  # small views are upscaled to 1024 later (generate.crop_to_object)
    return paths
