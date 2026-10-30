"""Prepare the views of an object for multi_image.py: take the pictures there are, make only the missing ones.

TRELLIS.2 has to guess what a picture doesn't show. multi_image.py can use several views, but often there
are only one or two. This script takes any number of pictures (or only a prompt), finds out from each file
name which side it shows, and lets Qwen-Image-Edit-2511 (local, through ComfyUI) draw only the sides that
are missing, always from the front picture:

    file name contains   side         file name contains        side
    front                0 deg        back                      180 deg
    right, side          90 deg       left                      270 deg
    (no word: the first picture is the front)

    (right = the camera moved to the right of the object, as the angle LoRA means it)

Without pictures, --prompt first makes the front picture. Two ways to make a missing side:

    angles   A (default): one Qwen edit per side with the multiple-angles LoRA ("<sks> back view ..."); every
                view is sharp, but they are made apart from each other, so they can disagree a little
    sheet    B: one Qwen edit that fills a turnaround sheet: the front in the first slot of a wide canvas, the
                missing sides drawn into the empty slots in one picture, so they agree better; smaller views,
                and Qwen sometimes draws the same side twice

Usage (TRELLIS.2 environment; ComfyUI is started if it isn't running):
    source activate.sh
    python scripts/models/make_views.py pictures/bee/bee-front.png pictures/bee/bee-side.png pictures/bee/bee-back.png
    python scripts/models/make_views.py pictures/knight/knight.jpg --way sheet
    python scripts/models/make_views.py --prompt "a small wooden treasure chest with iron bands"
    python scripts/models/multi_image.py outputs/views/bee-front_angles/view_*.png --resolution 1024 --steps 25

Writes outputs/views/<front picture>_<way>/view_<k>_<angle>deg.png (front first, as video_views.py does),
made.txt (which views were made by Qwen) and overview.jpg; for the sheet also sheet.png.
"""
import _paths  # noqa: F401  (the scripts of the other folders)
import argparse
import os
import re
import time

from PIL import Image

from edit_picture import comfy, start_comfyui, upload, wait_for_picture
from generate import OUTPUT_DIR, picture_name

ANGLES = {0: "front view", 90: "right side view", 180: "back view", 270: "left side view"}
WORDS = [("front", 0), ("back", 180), ("left", 270), ("right", 90), ("side", 90)]  # first match counts
GREY = (191, 191, 191)  # like edit_picture.py's background

FRONT_PROMPT = ("Replace the empty grey picture with {prompt}. The whole object seen straight from the front, "
                "centred, filling most of the picture, on a plain light grey background, even studio light, "
                "detailed 3D render.")

SHEET_PROMPT = ("Character turnaround sheet in one row of {n} equal slots on a plain light grey background. "
                "The first slot shows the object from the front. Fill the empty slots, left to right, with the "
                "same object seen from: {sides}. Exactly the same object, same size, same colours, materials "
                "and details in every slot, the whole object visible, orthographic, even studio light. "
                "Keep the first slot exactly as it is.")


def workflow(image_name, prompt, seed, angle_lora, megapixels=None):
    """Qwen-Image-Edit-2511 with the 4-step Lightning LoRA (and the angle LoRA), in ComfyUI's API format.
    megapixels: scale the picture to this size keeping its shape (for the wide sheet); None: Qwen's sizes."""
    lora_in = ["norm", 0]
    w = {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": "qwen_image_edit_2511_fp8mixed.safetensors", "weight_dtype": "default"}},
        "shift": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["unet", 0], "shift": 3.1}},
        "norm": {"class_type": "CFGNorm", "inputs": {"model": ["shift", 0], "strength": 1.0}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors", "type": "qwen_image"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
        "image": {"class_type": "LoadImage", "inputs": {"image": image_name}},
    }
    if angle_lora:
        w["angle"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": lora_in, "lora_name": "qwen-image-edit-2511-multiple-angles-lora.safetensors", "strength_model": 1.0}}
        lora_in = ["angle", 0]
    w["lora"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": lora_in, "lora_name": "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors", "strength_model": 1.0}}
    if megapixels:
        w["scaled"] = {"class_type": "ImageScaleToTotalPixels", "inputs": {"image": ["image", 0], "upscale_method": "lanczos", "megapixels": megapixels, "resolution_steps": 16}}
    else:
        w["scaled"] = {"class_type": "FluxKontextImageScale", "inputs": {"image": ["image", 0]}}
    w.update({
        "positive": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"clip": ["clip", 0], "prompt": prompt, "vae": ["vae", 0], "image1": ["scaled", 0]}},
        "negative": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"clip": ["clip", 0], "prompt": "", "vae": ["vae", 0], "image1": ["scaled", 0]}},
        "pos_ref": {"class_type": "FluxKontextMultiReferenceLatentMethod", "inputs": {"conditioning": ["positive", 0], "reference_latents_method": "index_timestep_zero"}},
        "neg_ref": {"class_type": "FluxKontextMultiReferenceLatentMethod", "inputs": {"conditioning": ["negative", 0], "reference_latents_method": "index_timestep_zero"}},
        "latent": {"class_type": "VAEEncode", "inputs": {"pixels": ["scaled", 0], "vae": ["vae", 0]}},
        "sample": {"class_type": "KSampler", "inputs": {"model": ["lora", 0], "seed": seed, "steps": 4, "cfg": 1.0, "sampler_name": "euler",
                                                        "scheduler": "simple", "positive": ["pos_ref", 0], "negative": ["neg_ref", 0],
                                                        "latent_image": ["latent", 0], "denoise": 1.0}},
        "decode": {"class_type": "VAEDecode", "inputs": {"samples": ["sample", 0], "vae": ["vae", 0]}},
        "save": {"class_type": "SaveImage", "inputs": {"images": ["decode", 0], "filename_prefix": "inpaint3d/views"}},
    })
    return w


def run(image, prompt, seed, angle_lora, megapixels=None):
    name = upload(image, "inpaint3d_views.png")
    job = comfy("/prompt", {"prompt": workflow(name, prompt, seed, angle_lora, megapixels), "client_id": "inpaint3d"})
    if job.get("node_errors"):
        raise RuntimeError(f"ComfyUI: {job['node_errors']}")
    return wait_for_picture(job["prompt_id"])


def on_grey(picture):
    """The picture as RGB; a transparent background becomes grey."""
    picture = picture.convert("RGBA")
    flat = Image.new("RGB", picture.size, GREY)
    flat.paste(picture, mask=picture.getchannel("A"))
    return flat


def side_of(path):
    """The angle a picture shows, from the words in its file name; None if there is no such word."""
    name = os.path.basename(path).lower()
    for word, angle in WORDS:
        if word in name:
            return angle
    return None


def sort_pictures(paths):
    """{angle: path}: each picture's side from its name, the first picture without a word is the front."""
    given = {}
    for path in paths:
        angle = side_of(path)
        if angle is None:
            angle = 0 if 0 not in given else None
        if angle is None or angle in given:
            raise SystemExit(f"{path}: which side does it show? Put front, back, left or right in its file name")
        given[angle] = path
    if 0 not in given:
        raise SystemExit("one picture must be the front (put 'front' in its file name)")
    return given


def by_angles(front, missing, seed):
    views = {}
    for angle in missing:
        print(f"  {angle} deg: {ANGLES[angle]}", flush=True)
        views[angle] = run(front, f"<sks> {ANGLES[angle]} eye-level shot medium shot", seed, angle_lora=True)
    return views, None


def by_sheet(front, missing, seed, megapixels):
    n = len(missing) + 1
    canvas = Image.new("RGB", (front.width * n, front.height), GREY)
    canvas.paste(front, (0, 0))
    prompt = SHEET_PROMPT.format(n=n, sides=", ".join(ANGLES[a] for a in missing))
    sheet = run(canvas, prompt, seed, angle_lora=False, megapixels=megapixels)
    slot = sheet.width / n
    views = {angle: sheet.crop((round(k * slot), 0, round((k + 1) * slot), sheet.height))
             for k, angle in enumerate(missing, start=1)}
    return views, sheet


def from_prompt(prompt, seed):
    """A front picture made from the prompt alone."""
    print(f"  front from the prompt: {prompt}", flush=True)
    return run(Image.new("RGB", (768, 1024), GREY), FRONT_PROMPT.format(prompt=prompt), seed, angle_lora=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pictures", nargs="*", help="the pictures there are (the side from the file name)")
    parser.add_argument("--prompt", help="without pictures: make the front from this text first")
    parser.add_argument("--way", choices=["angles", "sheet"], default="angles")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--megapixels", type=float, default=1.6, help="size of the whole sheet (--way sheet)")
    args = parser.parse_args()
    if not args.pictures and not args.prompt:
        parser.error("give pictures, or a --prompt")

    given = {a: on_grey(Image.open(p)) for a, p in sort_pictures(args.pictures).items()} if args.pictures else {}
    missing = [a for a in ANGLES if a not in given]
    name = picture_name(sort_pictures(args.pictures)[0]) if args.pictures else \
        re.sub(r"[^a-z0-9]+", "-", args.prompt.lower()).strip("-")[:40]
    folder = os.path.join(OUTPUT_DIR, "views", f"{name}_{args.way}")
    os.makedirs(folder, exist_ok=True)
    print(f"Given: {', '.join(ANGLES[a] for a in given) or 'nothing'}; "
          f"missing: {', '.join(ANGLES[a] for a in missing) or 'nothing'}", flush=True)

    start = time.time()
    made, sheet = {}, None
    if missing:
        started = start_comfyui()
        try:
            if 0 not in given:
                given[0] = made[0] = from_prompt(args.prompt, args.seed)
                missing.remove(0)
            if missing:
                print(f"Making {len(missing)} views ({args.way}) with Qwen-Image-Edit-2511...", flush=True)
                new, sheet = by_angles(given[0], missing, args.seed) if args.way == "angles" else \
                    by_sheet(given[0], missing, args.seed, args.megapixels)
                made.update(new)
        finally:
            if started is not None:
                started.terminate()  # we started it: stop it, so its GPU memory is free for TRELLIS.2
                started.wait(timeout=60)
            else:
                comfy("/free", {"unload_models": True, "free_memory": True})

    views = {a: made.get(a, given.get(a)) for a in ANGLES}
    for k, (angle, view) in enumerate(views.items()):
        view.save(os.path.join(folder, f"view_{k:02d}_{angle:03d}deg.png"))
    with open(os.path.join(folder, "made.txt"), "w") as f:
        f.write("".join(f"view_{k:02d}_{a:03d}deg.png made by Qwen ({args.way})\n" for k, a in enumerate(ANGLES) if a in made))
    if sheet is not None:
        sheet.save(os.path.join(folder, "sheet.png"))
    h = 400
    thumbs = [v.resize((round(v.width * h / v.height), h)) for v in views.values()]
    overview = Image.new("RGB", (sum(t.width for t in thumbs), h), "white")
    x = 0
    for t in thumbs:
        overview.paste(t, (x, 0))
        x += t.width
    overview.save(os.path.join(folder, "overview.jpg"), quality=90)
    print(f"Done in {time.time() - start:.0f} s ({len(made)} made): {folder}")


if __name__ == "__main__":
    main()
