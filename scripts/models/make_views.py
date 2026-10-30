"""Make the missing views of an object from one front picture, for multi_image.py.

TRELLIS.2 has to guess the back from a single picture. multi_image.py can use several views, but often
there is only one. This script lets Qwen-Image-Edit-2511 (local, through ComfyUI) draw the other sides:

    angles   A: one Qwen edit per view with the multiple-angles LoRA ("<sks> back view ..."), always
                made from the original front picture; every view is sharp, but the views are made apart
                from each other, so they can disagree a little
    sheet    B: one Qwen edit that fills a turnaround sheet: the front in the first slot of a wide canvas,
                Qwen draws the other sides into the empty slots, all in one picture, so they agree better;
                each view is smaller

Usage (TRELLIS.2 environment; ComfyUI is started if it isn't running):
    source activate.sh
    python scripts/models/make_views.py pictures/bee/bee-front.png --way angles
    python scripts/models/make_views.py pictures/bee/bee-front.png --way sheet --seed 2
    python scripts/models/multi_image.py outputs/views/bee-front_angles/view_*.png --resolution 1024 --steps 25

Writes outputs/views/<picture>_<way>/view_<k>_<angle>deg.png (front first, as video_views.py does) and
overview.jpg; for the sheet also sheet.png, the whole picture Qwen made.
"""
import _paths  # noqa: F401  (the scripts of the other folders)
import argparse
import os
import time

from PIL import Image

from edit_picture import comfy, start_comfyui, upload, wait_for_picture
from generate import OUTPUT_DIR, picture_name

ANGLES = {90: "right side view", 180: "back view", 270: "left side view"}  # the front (0) is the picture
GREY = (191, 191, 191)  # like edit_picture.py's background

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


def by_angles(front, seed):
    views = {}
    for angle, words in ANGLES.items():
        print(f"  {angle} deg: {words}", flush=True)
        views[angle] = run(front, f"<sks> {words} eye-level shot medium shot", seed, angle_lora=True)
    return views, None


def by_sheet(front, seed, megapixels):
    n = len(ANGLES) + 1
    canvas = Image.new("RGB", (front.width * n, front.height), GREY)
    canvas.paste(front, (0, 0))
    prompt = SHEET_PROMPT.format(n=n, sides=", ".join(ANGLES.values()))
    sheet = run(canvas, prompt, seed, angle_lora=False, megapixels=megapixels)
    slot = sheet.width / n
    views = {angle: sheet.crop((round(k * slot), 0, round((k + 1) * slot), sheet.height))
             for k, angle in enumerate(ANGLES, start=1)}
    return views, sheet


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("picture", help="the front picture of the object")
    parser.add_argument("--way", choices=["angles", "sheet"], default="angles")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--megapixels", type=float, default=1.6, help="size of the whole sheet (--way sheet)")
    args = parser.parse_args()

    front = on_grey(Image.open(args.picture))
    folder = os.path.join(OUTPUT_DIR, "views", f"{picture_name(args.picture)}_{args.way}")
    os.makedirs(folder, exist_ok=True)
    start = time.time()
    started = start_comfyui()
    try:
        print(f"Making {len(ANGLES)} views ({args.way}) with Qwen-Image-Edit-2511...", flush=True)
        views, sheet = by_angles(front, args.seed) if args.way == "angles" else by_sheet(front, args.seed, args.megapixels)
    finally:
        if started is not None:
            started.terminate()  # we started it: stop it, so its GPU memory is free for TRELLIS.2
            started.wait(timeout=60)
        else:
            comfy("/free", {"unload_models": True, "free_memory": True})

    views = {0: front, **views}
    for k, (angle, view) in enumerate(views.items()):
        view.save(os.path.join(folder, f"view_{k:02d}_{angle:03d}deg.png"))
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
    print(f"Done in {time.time() - start:.0f} s: {folder}")


if __name__ == "__main__":
    main()
