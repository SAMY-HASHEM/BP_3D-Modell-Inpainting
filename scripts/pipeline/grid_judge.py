"""Node 03: the grid judge. The local vision AI scores the model block by block; only bad blocks get re-inpainted.

The user's idea: instead of hand-made parts (head, bust, ...), the model's cube is cut into blocks (256^3 of
TRELLIS.2's 1024^3 grid: 4 x 4 x 4). For every block that has surface:

  1. camera     the camera of every picture the base model was made from (fit_camera, project_texture.py)
  2. views      which pictures see the block best (most visible pixels of its surface)
  3. judge      the picture's crop of the block next to the model's render of the same spot, from the same
                camera; the local AI (Qwen3.6-27B, Ollama) gives two scores from 0 to 100:
                  colour (a colour picture vs the textured render)
                  shape  (a grey sculpt picture vs a clay render; or the colour picture if there is no grey one)
  4. result     a block's score = its worst score; >= threshold (85): keep; below: bad (re-inpaint it)

Writes grid.json (every block: box in model coordinates, views, scores, bad or not) and heatmap.jpg (every
picture with each block's outline: green = good, red = bad, with its score).

Usage (TRELLIS.2 environment; Ollama with qwen3.6:27b):
    python scripts/pipeline/grid_judge.py model.glb --sorted 01_sort/.../sorted.json --out 03_grid/...
"""
import _paths  # noqa: F401  (the scripts of the other folders)

import argparse
import base64
import io
import json
import math
import os
import time
import urllib.request

import numpy as np
import nvdiffrast.torch as dr
import torch
from PIL import Image, ImageDraw

from project_texture import fit_camera, load_meshes, object_masks, render_meshes, to_clip

OLLAMA = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
JUDGE = "qwen3.6:27b"
MIN_PIXELS = 600  # a block needs at least this many visible pixels in a picture to be judged there

QUESTION = """The first picture is a crop of the artist's picture of a character. The second picture shows the same spot
of a 3D model of that character, from the same camera%s.
Compare them, for this spot only:
 - shape: are the forms the same (body, clothes, hair, folds, edges)? Holes, melted or broken parts, lumps?
 - colour: are the colours and patterns the same (skin, clothes, eyes, details)? Smears or black spots?
Lighting and background may differ; that does not count.
Answer only with JSON, no other text:
{"shape": 0-100, "colour": 0-100, "defects": ["..."], "why": "a few words"}"""


def encoded(image, size=512):
    image = image.convert("RGB")
    scale = size / max(image.size)
    image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=92)
    return base64.b64encode(buffer.getvalue()).decode()


BACKEND = {}  # "ollama" (Qwen3.6-27B, this PC) or "qwen3vl" (Qwen3-VL-8B through transformers, also on the server)


def ask(images, question):
    if BACKEND.get("kind") == "qwen3vl":
        import organise_pictures as ai
        if "model" not in BACKEND:
            BACKEND["model"], BACKEND["processor"] = ai.load(ai.MODELS[0], cpu=False)
        answer, _ = ai.ask(BACKEND["model"], BACKEND["processor"], [i.convert("RGB") for i in images], question, 300)
        return answer
    body = {"model": JUDGE, "stream": False, "think": False, "format": "json", "keep_alive": "10m",
            "options": {"temperature": 0}, "messages": [{"role": "user", "content": question,
                                                         "images": [encoded(i) for i in images]}]}
    request = urllib.request.Request(f"{OLLAMA}/api/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    text = json.load(urllib.request.urlopen(request, timeout=900))["message"]["content"]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def unload():
    if BACKEND.get("kind") == "qwen3vl":
        BACKEND.clear()
        torch.cuda.empty_cache()
        return
    body = {"model": JUDGE, "keep_alive": 0}
    try:
        urllib.request.urlopen(urllib.request.Request(f"{OLLAMA}/api/generate", data=json.dumps(body).encode(),
                                                      headers={"Content-Type": "application/json"}), timeout=60)
    except Exception:  # noqa: BLE001
        pass


def clay(ctx, meshes, cam, width, height):
    """A grey render of the shape alone (like a sculpt render): light from the camera's side."""
    image = torch.zeros((height, width, 3), device="cuda")
    best = torch.full((height, width), float("inf"), device="cuda")
    for m in meshes:
        clip = to_clip(m.verts, cam, width, height)[None]
        rast, _ = dr.rasterize(ctx, clip, m.faces, (height, width))
        normal, _ = dr.interpolate(m.normals[None], rast, m.faces)
        normal = torch.nn.functional.normalize(normal[0], dim=-1)
        # the light comes from the camera: the direction from the model's centre to the camera
        to_cam = torch.nn.functional.normalize(cam_direction(cam), dim=0)
        shade = (normal @ to_cam).abs().clamp(0, 1)[..., None] * 0.75 + 0.15
        z = torch.where(rast[0, ..., 3] > 0, rast[0, ..., 2], torch.full_like(best, float("inf")))
        nearer = z < best
        image = torch.where(nearer[..., None], shade.expand(-1, -1, 3), image)
        best = torch.where(nearer, z, best)
    return image


def cam_direction(cam):
    from project_texture import rotation
    r = rotation(cam["yaw"], cam["pitch"], cam["roll"])
    return r.T @ torch.tensor([0.0, 0.0, 1.0], device="cuda")  # camera +Z (towards the viewer) in model axes


def face_ids(ctx, meshes, cam, width, height):
    """Per pixel: which face (over all meshes, -1 = none) the camera sees there."""
    ids = torch.full((height, width), -1, dtype=torch.long, device="cuda")
    best = torch.full((height, width), float("inf"), device="cuda")
    offset = 0
    for m in meshes:
        clip = to_clip(m.verts, cam, width, height)[None]
        rast, _ = dr.rasterize(ctx, clip, m.faces, (height, width))
        z = torch.where(rast[0, ..., 3] > 0, rast[0, ..., 2], torch.full_like(best, float("inf")))
        nearer = z < best
        ids = torch.where(nearer, rast[0, ..., 3].long() - 1 + offset, ids)
        best = torch.where(nearer, z, best)
        offset += len(m.faces)
    return ids


def main():
    parser = argparse.ArgumentParser(description="Node 03: the local AI scores the model block by block")
    parser.add_argument("model")
    parser.add_argument("--sorted", required=True, help="node 01's sorted.json (the base model's pictures)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--blocks", type=int, default=4, help="blocks per side of the cube (4 = 256 of 1024)")
    parser.add_argument("--threshold", type=float, default=85)
    parser.add_argument("--judge", choices=["auto", "ollama", "qwen3vl"], default="auto",
                        help="auto: Ollama's qwen3.6:27b if it runs here, else Qwen3-VL-8B (e.g. on the uni server)")
    args = parser.parse_args()
    kind = args.judge
    if kind == "auto":
        try:
            tags = json.load(urllib.request.urlopen(f"{OLLAMA}/api/tags", timeout=3))
            kind = "ollama" if any(m["name"] == JUDGE for m in tags.get("models", [])) else "qwen3vl"
        except Exception:  # noqa: BLE001 (no Ollama here)
            kind = "qwen3vl"
    BACKEND["kind"] = kind
    print(f"judge: {JUDGE if kind == 'ollama' else 'Qwen3-VL-8B (transformers)'}", flush=True)
    os.makedirs(args.out, exist_ok=True)
    start = time.time()

    result = json.load(open(args.sorted))
    views = {v["name"]: v for v in result["views"]}
    base = result["lists"]["base"]
    names = base["both"] + base["colour_only"] + base["shape_only"]
    folder = os.path.dirname(args.sorted)

    scene, meshes = load_meshes(args.model)
    verts = torch.cat([m.verts for m in meshes])
    low, high = verts.min(0).values, verts.max(0).values
    size = float((high - low).max()) / args.blocks  # cubes: 1024 / 4 = 256 voxels of TRELLIS.2's grid
    counts = [max(1, math.ceil(float(high[a] - low[a]) / size - 1e-6)) for a in range(3)]
    centres = torch.cat([(m.verts[m.faces].mean(1)) for m in meshes])  # every face's centre
    cell = ((centres - low) / size).floor().long()
    for a in range(3):
        cell[:, a] = cell[:, a].clamp(0, counts[a] - 1)
    block_of_face = (cell[:, 0] * counts[1] + cell[:, 1]) * counts[2] + cell[:, 2]
    print(f"{len(centres):,} faces, blocks {counts} of {size:.3f} (model units)", flush=True)

    ctx = dr.RasterizeCudaContext()
    pictures = {n: Image.open(os.path.join(folder, views[n]["view"])).convert("RGB") for n in names}
    masks = dict(zip(names, object_masks([pictures[n] for n in names])))
    seen = {}  # block -> [(pixels, view name)]
    cams, renders = {}, {}
    for n in names:
        picture = pictures[n]
        cam, overlap = fit_camera(ctx, meshes, picture, masks[n])
        print(f"  camera of {n}: outline overlap {overlap:.2f}", flush=True)
        if overlap < 0.6:
            continue  # the camera doesn't fit: this picture can't judge
        w, h = picture.size
        ids = face_ids(ctx, meshes, cam, w, h)
        blocks = torch.where(ids >= 0, block_of_face[ids.clamp(min=0)], torch.full_like(ids, -1))
        colour, _ = render_meshes(ctx, meshes, cam, w, h)
        cams[n] = cam
        renders[n] = {"colour": Image.fromarray((colour.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)),
                      "clay": Image.fromarray((clay(ctx, meshes, cam, w, h).cpu().numpy() * 255).astype(np.uint8)),
                      "blocks": blocks.cpu().numpy()}
        values, number = torch.unique(blocks[blocks >= 0], return_counts=True)
        for b, c in zip(values.tolist(), number.tolist()):
            seen.setdefault(b, []).append((c, n))

    grid = []
    for b in sorted(seen):
        i, j, k = b // (counts[1] * counts[2]), (b // counts[2]) % counts[1], b % counts[2]
        box = [(low + size * torch.tensor([i, j, k], device="cuda")).tolist(),
               (low + size * torch.tensor([i + 1, j + 1, k + 1], device="cuda")).tolist()]
        entry = {"block": [i, j, k], "box": box, "scores": {}}
        is_colour = lambda n: views[n]["colours"] == "colour" and n not in base["shape_only"]
        ranked = sorted(seen[b], reverse=True)
        best_colour = next(((c, n) for c, n in ranked if is_colour(n) and c >= MIN_PIXELS), None)
        best_grey = next(((c, n) for c, n in ranked if not is_colour(n) and c >= MIN_PIXELS), None)
        for kind, pick in (("colour", best_colour), ("shape", best_grey)):
            if not pick:
                continue
            pixels, n = pick
            ys, xs = np.nonzero(renders[n]["blocks"] == b)
            pad = 0.15 * max(np.ptp(xs), np.ptp(ys)) + 8
            w, h = pictures[n].size
            crop = (int(max(0, xs.min() - pad)), int(max(0, ys.min() - pad)), int(min(w, xs.max() + pad)), int(min(h, ys.max() + pad)))
            ref = pictures[n].crop(crop)
            mine = renders[n]["colour" if kind == "colour" else "clay"].crop(crop)
            stem = f"b{i}{j}{k}_{kind}"
            ref.save(os.path.join(args.out, stem + "_picture.jpg"), quality=90)
            mine.save(os.path.join(args.out, stem + "_model.jpg"), quality=90)
            answer = ask([ref, mine], QUESTION % (", in colour" if kind == "colour" else " as a grey clay render"))
            entry["scores"][kind] = {"view": n, "pixels": pixels, "crop": crop, **answer}
            print(f"  block {i}{j}{k} {kind:6s} ({n[-24:]}, {pixels} px): shape {answer.get('shape')}, "
                  f"colour {answer.get('colour')}  {answer.get('why', '')}", flush=True)
        numbers = []
        if "colour" in entry["scores"]:
            numbers += [entry["scores"]["colour"].get("colour"), entry["scores"]["colour"].get("shape")]
        if "shape" in entry["scores"]:
            numbers += [entry["scores"]["shape"].get("shape")]
        numbers = [x for x in numbers if isinstance(x, (int, float))]
        if not numbers:
            continue
        entry["score"] = min(numbers)
        entry["bad"] = entry["score"] < args.threshold
        # what to redo: the shape too, or only the colours
        shapes = [s.get("shape") for s in entry["scores"].values() if isinstance(s.get("shape"), (int, float))]
        entry["redo"] = ("shape" if shapes and min(shapes) < args.threshold else "colour") if entry["bad"] else None
        grid.append(entry)
    unload()

    json.dump({"model": args.model, "threshold": args.threshold, "block_size": size, "counts": counts,
               "judge": JUDGE if kind == "ollama" else "Qwen/Qwen3-VL-8B-Instruct", "blocks": grid, "seconds": round(time.time() - start)},
              open(os.path.join(args.out, "grid.json"), "w"), indent=1)

    # heat map: every judged picture with each block's outline and score
    tiles = []
    for n in cams:
        image = Image.blend(pictures[n], renders[n]["colour"], 0.5)
        draw = ImageDraw.Draw(image)
        for e in grid:
            for s in e["scores"].values():
                if s["view"] == n:
                    x0, y0, x1, y1 = s["crop"]
                    colour = (60, 200, 80) if not e["bad"] else (230, 60, 50)
                    draw.rectangle((x0, y0, x1, y1), outline=colour, width=3)
                    draw.text((x0 + 4, y0 + 2), f"{e['score']}", fill=colour)
        image.thumbnail((520, 1100))
        tiles.append(image)
    if tiles:
        sheet = Image.new("RGB", (sum(t.width for t in tiles), max(t.height for t in tiles)), (40, 40, 44))
        x = 0
        for t in tiles:
            sheet.paste(t, (x, 0))
            x += t.width
        sheet.save(os.path.join(args.out, "heatmap.jpg"), quality=90)
    bad = [e for e in grid if e["bad"]]
    print(f"{len(grid)} blocks judged, {len(bad)} below {args.threshold}: "
          + ", ".join(f"{''.join(map(str, e['block']))} ({e['score']}, {e['redo']})" for e in bad), flush=True)
    print(f"Result: {os.path.join(args.out, 'grid.json')}", flush=True)
    print(f"Done in {time.time() - start:.0f} s", flush=True)


if __name__ == "__main__":
    main()
