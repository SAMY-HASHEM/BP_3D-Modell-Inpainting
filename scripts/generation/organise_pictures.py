"""Let a local vision AI organise the pictures (and videos) of an object before it is generated in 3D.

People collect pictures of an object from anywhere: full views and close-ups, sheets with two views side
by side, colour-ID renders, pictures with a pedestal or a logo. TRELLIS.2 needs clean full views of the
same pose, the front first. This script lets a local vision-language model (Qwen3-VL) decide, instead of
a person:

  a video              the AI watches frames sampled evenly over it: is it a turntable, where does one full
                       turn end, which frame is the front, which are blurred; the frames it keeps go on
                       like pictures
  pass 1, every picture    how many figures it shows, a box around each one, which side each shows
                           (front, back, left, ...), full body or close-up, the pose, the background, colour
                           or grey, extras (pedestal, text, weapon); every figure is cut out with its box
  pass 2, every object     all full-body cut-outs together: which one is the front, which views belong to
                           the same pose and help the 3D model, which to leave out and why, and what kind of
                           object it is (symmetric, thin parts, how detailed)

Writes into --out: organised.json (all answers) and crops/<object>_<n>_<side>.png. The answers are the
AI's own; nothing here knows what the pictures show. They are kept with the test results so that later
the settings can be learned from them.

Usage (TRELLIS.2 environment; the model is downloaded into local_ai_models/ the first time):
    python scripts/generation/organise_pictures.py pictures/ninja pictures/monster --out outputs/organised
    python scripts/generation/organise_pictures.py pictures/ninja --model Qwen/Qwen3-VL-2B-Instruct --cpu   # small test
"""
import argparse
import glob
import json
import os
import re
import time

import torch
from PIL import Image

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_HUB_CACHE", os.path.join(PROJECT_DIR, "local_ai_models"))
MODELS = ("Qwen/Qwen3-VL-8B-Instruct", "Qwen/Qwen3-VL-2B-Instruct", "Qwen/Qwen2.5-VL-7B-Instruct")  # first = default
BACKUP = "Qwen/Qwen2.5-VL-7B-Instruct"  # for an older transformers that doesn't know Qwen3-VL yet
SIDES = ("front", "front-left", "left", "back-left", "back", "back-right", "right", "front-right", "top", "bottom", "unknown")
PICTURE_TYPES = (".png", ".jpg", ".jpeg", ".webp")
VIDEO_TYPES = (".mp4", ".mov", ".webm", ".avi", ".mkv", ".m4v")

VIDEO = """These %d pictures are frames of one video, in time order, numbered 0 to %d (evenly spaced over the video).
Answer only with JSON, no other text:
{"turntable": true | false,      // the object turns around (or the camera goes around it)
 "turns": how many full turns the video shows, e.g. 0.5, 1 or 2,
 "turn_end": the number of the frame where the first full turn is complete (the view looks like frame 0 again), or null,
 "front": the number of the frame that shows the front best,
 "blurred": [numbers of frames that are blurred or show mostly something else],
 "why": "a few words"}"""

PASS_1 = """Look at this picture. It may show one or several figures or objects (for example the same character
from two sides next to each other), or only a close-up of a part.

Answer only with JSON, no other text:
{"object": "a short name of what is shown, e.g. ninja girl, winged monster, robot",
 "background": "light" | "dark" | "coloured" | "busy",
 "colours": "colour" | "grey",
 "extras": [things that are not part of the object: "pedestal", "text", "logo", "floor", "shadow", "frame", ...],
 "figures": [
   {"box": [x0, y0, x1, y1],            // the whole figure, coordinates from 0 to 1000 of the picture
    "side": one of %s,                  // which side of the object we see
    "full_body": true | false,          // false for a close-up of only a part
    "pose": "neutral" | "action",       // neutral = standing still, T- or A-pose
    "id_render": true | false,          // true for a flat-coloured part-ID / segmentation picture
    "useful_for_3d": true | false,      // would this view help to build the whole object in 3D?
    "why": "a few words"}]}""" % "|".join(f'"{s}"' for s in SIDES)

PASS_2 = """These %d pictures (numbered 0 to %d in this order) are cut-outs of the same object: %s.
A 3D model will be made from some of them. The views used together must show the same object in the same pose;
the first one must be the best front view. Close-ups, other poses, other versions and part-ID renders confuse it.

Answer only with JSON, no other text:
{"front": the number of the best front view,
 "views": [numbers of the views to use together, front first, at most 4, each from a different side],
 "left_out": {"number": "why", ...},
 "symmetric": true | false,             // left and right side mirror each other
 "thin_parts": true | false,            // wings, antennae, blades, hair strands, fingers ...
 "detail": "low" | "medium" | "high",   // how much fine surface detail
 "kind": "character" | "creature" | "vehicle" | "prop" | "building" | "other"}"""


def load(model_name, cpu):
    from transformers import AutoModelForImageTextToText, AutoProcessor
    print(f"Loading {model_name} ({'CPU' if cpu else 'GPU'})...", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(model_name, dtype=torch.float32 if cpu else torch.bfloat16)
    model = model.to("cpu" if cpu else "cuda").eval()
    return model, AutoProcessor.from_pretrained(model_name)


def ask(model, processor, images, question, max_tokens=1200):
    """One question about one or several pictures; returns the answer's JSON (or {} if there is none)."""
    content = [{"type": "image", "image": im} for im in images] + [{"type": "text", "text": question}]
    inputs = processor.apply_chat_template([{"role": "user", "content": content}], tokenize=True,
                                           add_generation_prompt=True, return_dict=True, return_tensors="pt")
    inputs = inputs.to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    text = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    return parse(text), text


def parse(text):
    """The JSON in a model answer (also inside ```json fences, with // comments or trailing commas)."""
    text = re.sub(r"```(?:json)?", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        return {}
    body = re.sub(r"//[^\n]*", "", text[start:end + 1])
    body = re.sub(r",\s*([}\]])", r"\1", body)
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return {}


def shrink(image, size):
    image = image.copy()
    image.thumbnail((size, size))
    return image


def to_pixels(box, width, height, shown):
    """The model's box (0-1000 of the picture; some answers use pixels of the picture it saw) in pixels
    of the original picture, a little larger so nothing is cut off; None if it makes no sense."""
    try:
        x0, y0, x1, y1 = (float(v) for v in box)
    except (TypeError, ValueError):
        return None
    scale = 1000.0 if max(x0, y0, x1, y1) <= 1000 else float(max(shown))
    x0, x1 = sorted((x0 / scale * width, x1 / scale * width))
    y0, y1 = sorted((y0 / scale * height, y1 / scale * height))
    if x1 - x0 < 0.05 * width or y1 - y0 < 0.05 * height:
        return None
    pad = 0.04 * max(x1 - x0, y1 - y0)
    return (max(0, int(x0 - pad)), max(0, int(y0 - pad)), min(width, int(x1 + pad)), min(height, int(y1 + pad)))


def sample_frames(video, count, out, name):
    """`count` frames evenly spaced over the whole video (no choice made here), saved to out/frames/."""
    import cv2
    capture = cv2.VideoCapture(video)
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    paths = []
    for k in range(count):
        capture.set(cv2.CAP_PROP_POS_FRAMES, round(k * (total - 1) / max(1, count - 1)))
        ok, frame = capture.read()
        if ok:
            path = os.path.join(out, "frames", f"{name}_f{k:02d}.jpg")
            cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            paths.append(path)
    return paths


def look_at_video(model, processor, video, out, name, count):
    """The AI watches the video's frames: turntable or not, where one turn ends, the front, blurred frames.
    Returns the frames it keeps and its answer."""
    frames = sample_frames(video, count, out, name)
    images = [shrink(Image.open(f).convert("RGB"), 384) for f in frames]
    answer, raw = ask(model, processor, images, VIDEO % (len(images), len(images) - 1))
    print(f"  {os.path.basename(video)}: {json.dumps(answer)}", flush=True)
    keep = list(range(len(frames)))
    end = answer.get("turn_end")
    if answer.get("turntable") and isinstance(end, int) and 2 <= end < len(frames):
        keep = keep[:end]  # one full turn; the frames after it repeat it
    blurred = {i for i in answer.get("blurred", []) if isinstance(i, int)} if isinstance(answer.get("blurred"), list) else set()
    keep = [i for i in keep if i not in blurred] or keep
    return [frames[i] for i in keep], {"file": os.path.basename(video), "video": True, "answer": answer, "raw": raw,
                                       "frames": [os.path.relpath(f, out) for f in frames],
                                       "kept": [os.path.relpath(frames[i], out) for i in keep]}


def evenly(items, most):
    """At most `most` items, evenly spaced (keeps the order)."""
    if len(items) <= most:
        return items
    return [items[round(k * (len(items) - 1) / (most - 1))] for k in range(most)]


def organise(model, processor, folder, out, size, frames=24):
    name = os.path.basename(os.path.normpath(folder)).replace(" ", "_")
    everything = sorted(glob.glob(os.path.join(folder, "*")))
    files = [f for f in everything if f.lower().endswith(PICTURE_TYPES)]
    videos = []
    for video in (f for f in everything if f.lower().endswith(VIDEO_TYPES)):
        kept, entry = look_at_video(model, processor, video, out, name, frames)
        files += kept
        videos.append(entry)
    pictures, crops = [], []
    for path in files:
        image = Image.open(path).convert("RGB")
        shown = shrink(image, size)
        answer, raw = ask(model, processor, [shown], PASS_1)
        print(f"  {os.path.basename(path)}: {json.dumps(answer)[:300]}", flush=True)
        source = os.path.relpath(path, out) if path.startswith(os.path.join(out, "frames")) else os.path.basename(path)
        entry = {"file": source, "size": list(image.size), "answer": answer, "raw": raw, "crops": []}
        for k, figure in enumerate(answer.get("figures", []) if isinstance(answer.get("figures"), list) else []):
            box = to_pixels(figure.get("box"), image.width, image.height, shown.size) if isinstance(figure, dict) else None
            if box is None:
                continue
            side = figure.get("side") if figure.get("side") in SIDES else "unknown"
            crop = os.path.join("crops", f"{name}_{len(crops):02d}_{side}.png")
            image.crop(box).save(os.path.join(out, crop))
            entry["crops"].append(crop)
            crops.append({"crop": crop, "from": source, "box": box, **figure, "side": side})
        pictures.append(entry)

    # pass 2: all full-body cut-outs together
    candidates = evenly([c for c in crops if c.get("full_body") and not c.get("id_render")], 10)
    choice = {}
    if candidates:
        thing = next((p["answer"].get("object") for p in pictures if p["answer"].get("object")), name)
        images = [shrink(Image.open(os.path.join(out, c["crop"])).convert("RGB"), 448) for c in candidates]
        choice, raw = ask(model, processor, images, PASS_2 % (len(images), len(images) - 1, thing))
        print(f"  choice: {json.dumps(choice)}", flush=True)
        choice["raw"] = raw
        pick = lambda i: candidates[i]["crop"] if isinstance(i, int) and 0 <= i < len(candidates) else None
        views = [pick(i) for i in choice.get("views", []) if pick(i)]
        front = pick(choice.get("front"))
        if front and front in views:
            views.remove(front)
        choice["front_crop"] = front or (views[0] if views else candidates[0]["crop"])
        choice["view_crops"] = [choice["front_crop"]] + [v for v in views if v != choice["front_crop"]][:3]
    return {"object": name, "videos": videos, "pictures": pictures, "crops": crops,
            "candidates": [c["crop"] for c in candidates], "choice": choice}


def main():
    parser = argparse.ArgumentParser(description="Let a local vision AI organise an object's pictures for TRELLIS.2")
    parser.add_argument("folders", nargs="+", help="one folder per object: pictures and/or videos")
    parser.add_argument("--frames", type=int, default=24, help="frames the AI looks at per video (default 24)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default=MODELS[0], choices=MODELS)
    parser.add_argument("--size", type=int, default=1024, help="longest side of the pictures the AI sees")
    parser.add_argument("--cpu", action="store_true", help="run without a GPU (slow; for testing the small model)")
    parser.add_argument("--zip", action="store_true", help="also pack --out into one .zip (for the cloud worker)")
    args = parser.parse_args()
    os.makedirs(os.path.join(args.out, "crops"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "frames"), exist_ok=True)
    start = time.time()
    try:
        model, processor = load(args.model, args.cpu)
    except (KeyError, ValueError, ImportError) as e:  # e.g. "model type qwen3_vl not recognized"
        print(f"{args.model} can't be loaded here ({type(e).__name__}: {str(e)[:150]}); using {BACKUP}", flush=True)
        args.model = BACKUP
        model, processor = load(args.model, args.cpu)
    result = {"model": args.model, "objects": []}
    for folder in args.folders:
        print(f"Organising {folder}", flush=True)
        result["objects"].append(organise(model, processor, folder, args.out, args.size, args.frames))
        with open(os.path.join(args.out, "organised.json"), "w") as f:
            json.dump(result, f, indent=1)
    result["seconds"] = round(time.time() - start)
    with open(os.path.join(args.out, "organised.json"), "w") as f:
        json.dump(result, f, indent=1)
    if args.zip:
        import shutil
        archive = shutil.make_archive(os.path.normpath(args.out), "zip", args.out)
        print(f"Result: {archive}", flush=True)
    print(f"Done in {time.time() - start:.0f} s", flush=True)


if __name__ == "__main__":
    main()
