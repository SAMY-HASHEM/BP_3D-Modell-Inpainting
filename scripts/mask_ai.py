"""Finds a part (e.g. "helmet") in pictures of a model and writes one black-and-white mask per picture.

Used by the Blender add-on's "Find part" button: Blender takes pictures of the space box from several
sides, this script finds the part in each picture, and Blender turns the masks into a 3D selection (each
face is voted on by the pictures that see it). Runs in its own environment (./install_mask.sh), on the
GPU if there is one (about 4 GB of VRAM), else on the CPU.

    python scripts/mask_ai.py <folder>          # a folder with views.json and the pictures it names
    python scripts/mask_ai.py --download        # only download the models (into models/mask)

views.json:  {"word": "helmet", "views": [{"picture": "view_00.png", "clicks": [[x, y], ...]}, ...]}
             (clicks are pixels where the user's click is seen in that picture, if it is seen there)
Writes:      <folder>/masks.npz   one yes/no picture per view, named like its picture ("view_00"); a
                                  picture where nothing was found has none (it doesn't vote)
             <folder>/overview.jpg  all pictures with what was found in red

How the part is found in one picture:
  1. Grounding DINO reads the word and puts boxes around what matches it (only boxes it is sure about,
     and not around nearly the whole picture: for "helmet" it also boxes the whole statue wearing it)
  2. SAM 2.1 cuts out the exact outline in each box. A click keeps only the boxes it lies in; if it lies
     in none, the thing under the click is cut out instead.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from PIL import Image

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_DIR, "models", "mask")
DINO = "IDEA-Research/grounding-dino-base"
SAM = "facebook/sam2.1-hiera-large"
BOX_THRESHOLD = 0.25   # how sure Grounding DINO must be that a box shows the word
TEXT_THRESHOLD = 0.25  # ... and that the word (not another one in the sentence) is meant
WHOLE_PICTURE = 0.5    # a box over more than this share of the picture means "the whole thing": ignored


def pick_device(cpu_only):
    if not cpu_only and torch.cuda.is_available():
        try:
            torch.ones(1, device="cuda").add_(1)  # a GPU this PyTorch doesn't support fails here
            return "cuda"
        except Exception as e:
            print(f"GPU not usable ({str(e).splitlines()[0]}), using the CPU", flush=True)
    return "cpu"


class PartFinder:
    def __init__(self, device):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, Sam2Model, Sam2Processor
        self.device = device
        self.dino_processor = AutoProcessor.from_pretrained(DINO, cache_dir=MODELS_DIR)
        self.dino = AutoModelForZeroShotObjectDetection.from_pretrained(DINO, cache_dir=MODELS_DIR).to(device).eval()
        self.sam_processor = Sam2Processor.from_pretrained(SAM, cache_dir=MODELS_DIR)
        self.sam = Sam2Model.from_pretrained(SAM, cache_dir=MODELS_DIR).to(device).eval()

    @torch.no_grad()
    def boxes(self, picture, word):
        """Boxes (x0, y0, x1, y1 in pixels) around what matches the word, most certain first."""
        inputs = self.dino_processor(images=picture, text=[[word]], return_tensors="pt").to(self.device)
        result = self.dino_processor.post_process_grounded_object_detection(
            self.dino(**inputs), inputs.input_ids, threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD,
            target_sizes=[picture.size[::-1]])[0]
        boxes, scores = result["boxes"].cpu().numpy(), result["scores"].cpu().numpy()
        share = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) / (picture.size[0] * picture.size[1])
        keep = np.flatnonzero(share <= WHOLE_PICTURE)
        keep = keep[np.argsort(-scores[keep])]
        return boxes[keep], scores[keep]

    @torch.no_grad()
    def cut_out(self, picture, boxes=None, clicks=None):
        """The outline of what is in the boxes (or under the clicks), as one yes/no picture."""
        if boxes is not None:
            inputs = self.sam_processor(images=picture, input_boxes=[[list(map(float, b)) for b in boxes]],
                                        return_tensors="pt")
            outputs = self.sam(**inputs.to(self.device), multimask_output=False)
        else:  # one click can mean a small piece or the whole thing: take SAM's most confident of three
            inputs = self.sam_processor(images=picture, input_points=[[[list(map(float, c))] for c in clicks]],
                                        input_labels=[[[1] for _ in clicks]], return_tensors="pt")
            outputs = self.sam(**inputs.to(self.device), multimask_output=True)
            best = outputs.iou_scores.argmax(-1, keepdim=True)  # per click
            outputs.pred_masks = torch.take_along_dim(outputs.pred_masks, best[..., None, None], dim=2)
        masks = self.sam_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu())[0]
        return masks.reshape(-1, *masks.shape[-2:]).any(0).numpy()

    def find(self, picture, word, clicks):
        """The part in one picture: a yes/no picture, and a short note on how it was found."""
        boxes = np.zeros((0, 4))
        if word:
            boxes, scores = self.boxes(picture, word)
        if clicks:
            inside = [b for b in boxes if any(b[0] <= x <= b[2] and b[1] <= y <= b[3] for x, y in clicks)]
            if inside:
                return self.cut_out(picture, boxes=inside), f"{len(inside)} box(es) with the click"
            return self.cut_out(picture, clicks=clicks), "the thing under the click"
        if len(boxes):
            return self.cut_out(picture, boxes=boxes), f"{len(boxes)} box(es), best {scores[0]:.2f}"
        return None, "not found (this picture doesn't vote)"


def load_picture(path):
    """A picture as RGB; a transparent background (as Blender renders it) becomes grey."""
    picture = Image.open(path)
    if picture.mode == "RGBA":
        picture = Image.alpha_composite(Image.new("RGBA", picture.size, (128, 128, 128, 255)), picture)
    return picture.convert("RGB")


def overview(pictures, masks, path, size=320):
    """All pictures side by side, with what was found in red."""
    if not pictures:
        return
    tiles = []
    for picture, mask in zip(pictures, masks):
        rgb = np.asarray(picture.convert("RGB"), np.float32)
        rgb[mask] = rgb[mask] * 0.45 + np.array([255, 40, 40]) * 0.55
        tiles.append(Image.fromarray(rgb.astype(np.uint8)).resize((size, size)))
    columns = min(4, len(tiles))
    sheet = Image.new("RGB", (columns * size, -(-len(tiles) // columns) * size), (40, 40, 40))
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % columns) * size, (i // columns) * size))
    sheet.save(path, quality=90)


def main():
    parser = argparse.ArgumentParser(description="Find a part (e.g. 'helmet') in pictures of a model")
    parser.add_argument("folder", nargs="?", help="folder with views.json and the pictures")
    parser.add_argument("--cpu", action="store_true", help="don't use the GPU")
    parser.add_argument("--download", action="store_true", help="only download the models")
    args = parser.parse_args()
    if not args.download and not args.folder:
        parser.error("give the folder with views.json (or --download)")

    start = time.time()
    device = pick_device(args.cpu)
    print(f"Loading the mask AI on the {'GPU' if device == 'cuda' else 'CPU'}...", flush=True)
    finder = PartFinder(device)
    if args.download:
        print(f"Models ready in {MODELS_DIR}")
        return

    with open(os.path.join(args.folder, "views.json")) as f:
        job = json.load(f)
    word = job.get("word", "").strip().lower()
    pictures, masks, found = [], [], {}
    for i, view in enumerate(job["views"]):
        picture = load_picture(os.path.join(args.folder, view["picture"]))
        mask, note = finder.find(picture, word, view.get("clicks") or [])
        pictures.append(picture)
        masks.append(mask if mask is not None else np.zeros(picture.size[::-1], bool))
        if mask is not None:
            found[os.path.splitext(view["picture"])[0]] = mask
            note += f", {mask.mean():.0%} of the picture"
        print(f"picture {i + 1}/{len(job['views'])}: {note}", flush=True)
    np.savez_compressed(os.path.join(args.folder, "masks.npz"), **found)
    overview(pictures, masks, os.path.join(args.folder, "overview.jpg"))
    print(f"Done in {time.time() - start:.0f} s: {os.path.join(args.folder, 'masks.npz')}", flush=True)


if __name__ == "__main__":
    main()
