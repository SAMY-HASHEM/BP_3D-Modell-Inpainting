"""Node 05 of the inpainting flow: the local judge of an edit (e.g. the crown on the girl's head).

Every candidate (one per seed or setting) is rendered from the same four straight sides as the edit's pictures
(make_edit.py: before, after, 2D mask per side), so every pixel compares. Three things are MEASURED per side:

  kept    outside the edit's area (the 3D area's cubes seen from that side, and the 2D mask) the model must be the original: its outline (100 - 300 x the share of pixels that
          appeared or went) and its colours (100 - 1.5 x the mean colour difference, CIE Lab); the lower one counts
  part    inside the 2D mask the new part must be there: the share of the mask where the candidate differs from
          the original (100 - 300 x the share that stayed the same, 6 pixels of tolerance)
  colour  inside the 2D mask: 100 - 1.5 x the difference of the mean colours to the AFTER picture (it only counts
          after the colours step; before it the colours are whatever TRELLIS.2 made along the way)

Two more numbers are only written down, to compare with the papers (VoxHammer's Edit3D-Bench, EditVerse3D measure the
unedited region this way): chamfer = the mean distance between the original's and the candidate's surface outside
the 3D area, as a share of the model's size (VoxHammer reports 0.012, EditVerse3D 0.002-0.005 in their own units),
and psnr = the masked PSNR of the renders outside the area (VoxHammer 41.7 dB, EditVerse3D 35.5 dB).

Then the local vision AI (Qwen3.6-27B, Ollama) looks at one candidate at a time, the AFTER picture next to the
render of the same side and of the next side, lists defects (floating, broken, melted, in the wrong place) and
gives 1-10. The best candidate has the highest measured score (the lower of kept and part); the AI's score breaks
ties within 5 points and is written down next to it. Nothing is chosen by hand, and Claude does not choose.

    python scripts/pipeline/edit_judge.py --edit outputs/edits/girl_crown --original girl.glb \
        --candidates seed1.glb seed2.glb --out folder [--step colours] [--no-ai]

Writes judge.json (all numbers, "best", "verdict": pass / redo) and compare.jpg.
"""
import _paths  # noqa: F401  (the scripts of the other folders)

import argparse
import json
import os
import subprocess

import numpy as np
from PIL import Image, ImageDraw

PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BLENDER = os.environ.get("BLENDER", os.path.expanduser("~/Applications/blender/blender"))
GREY = (190, 190, 192)

QUESTION = """The first picture shows what was wanted: %s with %s put on it (a rough collage, so its edges are hard).
The second picture shows the 3D model that was made, from the same side; the third picture shows it from the next side.

Judge only the new part, and whether the rest of the object looks the same as in the first picture. What the object
itself lacks in all pictures (e.g. it is a bust without legs, or it has no face) is NOT a defect.
Is the new part there, in the same place as in the first picture, complete, and attached to the object (not
floating above it, not sunk into it)? First list the defects of the new part: missing or only half there, floating
pieces, a broken or melted shape, holes, the wrong place or the wrong size, and any damage to the rest of the object
compared with the first picture. Then give a score: 10 = the part is there as wanted and nothing else changed;
1 = the part is missing or unusable. Clear defects make the score low.
Answer only with JSON, no other text:
{"part_is_there": true or false, "defects": ["..."], "score": a number from 1 to 10, "why": "a few words"}"""

# what this step makes: the shape steps leave the surface for the colours step, so grey or patchy colours are not their fault
STEP_NOTE = {"shape": "This step makes only the SHAPE: the colours are painted in a later step, so grey, patchy, noisy or wrong\n"
                      "colours and textures are NOT defects. Judge the shape only.\n",
             "details": "This step makes only the SHAPE and its details: the colours are painted in a later step, so grey, patchy,\n"
                        "noisy or wrong colours and textures are NOT defects. Judge the shape only.\n",
             "colours": "This step paints the colours: judge the shape and the colours.\n"}


def lab(rgb):
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    m = np.array([[.4124, .3576, .1805], [.2126, .7152, .0722], [.0193, .1192, .9505]])
    xyz = linear @ m.T / np.array([.9505, 1, 1.089])
    f = np.where(xyz > .00886, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def on_grey(view):
    plate = Image.new("RGBA", view.size, GREY + (255,))
    plate.alpha_composite(view.convert("RGBA"))
    return np.asarray(plate.convert("RGB"), np.float32) / 255


def wide(mask, pixels):
    from scipy import ndimage
    return ndimage.binary_dilation(mask, iterations=pixels) if pixels > 0 else ndimage.binary_erosion(mask, iterations=-pixels)


def area_seen(points, info, index, voxels=1.5):
    """The edit's 3D area (area.npy, the model's glTF axes) seen from one side: the pixels its cubes cover."""
    from scipy import ndimage
    size = info["size"]
    to_gltf = lambda p: np.array([p[0], p[2], -p[1]])
    origin, right = to_gltf(info["center"]), to_gltf(info["views"][index]["right"])
    per_pixel = info["span"] / size
    u = ((points - origin) @ right / per_pixel + size / 2).astype(int)
    v = (size / 2 - (points[:, 1] - origin[1]) / per_pixel).astype(int)
    good = (u >= 0) & (u < size) & (v >= 0) & (v < size)
    seen = np.zeros((size, size), bool)
    seen[v[good], u[good]] = True
    cube = info["span"] / 1.4 / 64 / per_pixel  # one of TRELLIS.2's 64 voxels, in pixels
    return ndimage.binary_dilation(seen, iterations=max(1, int(voxels * cube)))


def measure_side(original, candidate, mask, after, area):
    """One side: the original's render, the candidate's render (RGBA, the same camera), the 2D mask (or None), the
    AFTER picture (or None) and the edit's area seen from this side."""
    a_on, b_on = np.asarray(original)[..., 3] > 128, np.asarray(candidate)[..., 3] > 128
    a, b = lab(on_grey(original)), lab(on_grey(candidate))
    difference = np.linalg.norm(a - b, axis=-1)
    outside = ~(wide(mask, 12) | area)
    # kept: outside the mask nothing may appear or go, and the colours stay
    appeared = (b_on & ~wide(a_on, 3) & outside).sum() / max(1, (a_on & outside).sum())
    gone = (a_on & ~wide(b_on, 3) & outside).sum() / max(1, (a_on & outside).sum())
    both = a_on & b_on & outside
    colour_change = float(difference[both].mean()) if both.any() else 0.0
    kept = min(100 - 300 * max(appeared, gone), 100 - 1.5 * colour_change)
    # part: inside the mask the candidate differs from the original (new surface, or another colour)
    wanted = wide(mask, -4) if wide(mask, -4).sum() > 50 else mask
    changed = wide((a_on != b_on) | (difference > 12), 6)
    same = (wanted & ~changed).sum() / max(1, wanted.sum())
    part = 100 - 300 * same
    x, y = on_grey(original), on_grey(candidate)
    error = float(((x - y) ** 2)[outside].mean())
    psnr = round(float(min(99.0, 10 * np.log10(1 / max(error, 1e-10)))), 1)  # masked PSNR, as the papers report it
    result = {"kept": round(float(np.clip(kept, 0, 100))), "part": round(float(np.clip(part, 0, 100))), "psnr": psnr,
              "appeared": round(float(appeared), 4), "gone": round(float(gone), 4),
              "colour_change": round(colour_change, 1), "unchanged_in_mask": round(float(same), 3)}
    if after is not None:  # colours: the mean colour inside the mask next to the AFTER picture's
        wish = lab(np.asarray(after.convert("RGB"), np.float32) / 255)
        there = wanted & b_on
        if there.sum() > 50:
            away = float(np.linalg.norm(wish[there].mean(0) - b[there].mean(0)))
            result.update(colour=round(float(np.clip(100 - 1.5 * away, 0, 100))), colour_difference=round(away, 1))
        # like: how close the candidate is to the AFTER picture around the change. Only fair when AFTER is a render of
        # the wanted model (place_part.py), not a collage: where there is something in one and nothing in the other,
        # and the colour difference pixel by pixel where both have something. `part` only says that something changed:
        # a brown plate where a toast was got 0, a torn slab 80 (the toaster robot's book, Nov 16)
        around = wide(mask, 8)
        if around.sum() > 50:
            wish_on = np.abs(np.asarray(after.convert("RGB"), int) - np.array(GREY)).sum(-1) > 24
            wrong = float(((wish_on != b_on) & around).sum() / around.sum())
            both = wish_on & b_on & around
            apart = float(np.linalg.norm(wish[both] - b[both], axis=-1).mean()) if both.any() else 0.0
            result.update(like=round(float(np.clip(min(100 - 300 * wrong, 100 - 1.5 * apart), 0, 100))),
                          like_shape=round(float(np.clip(100 - 300 * wrong, 0, 100))), like_colour=round(float(np.clip(100 - 1.5 * apart, 0, 100))))
    return result


def chamfer_kept(original, candidate, points, count=200_000):
    """The surface outside the edit's 3D area: mean distance original <-> candidate, as a share of the model's size."""
    import trimesh
    from scipy.spatial import cKDTree
    a, b = (trimesh.load(path, force="mesh") for path in (original, candidate))
    size = float(np.ptp(np.asarray(a.vertices), 0).max())
    area, reach = cKDTree(points), 2.5 * size / 64  # the area's cubes and a little around them don't count
    samples = []
    for mesh in (a, b):
        surface = trimesh.sample.sample_surface(mesh, count, seed=0)[0]
        samples.append(surface[area.query(surface)[0] > reach])
    if min(len(s) for s in samples) < 100:
        return None
    one, two = cKDTree(samples[1]).query(samples[0])[0], cKDTree(samples[0]).query(samples[1])[0]
    return round(float((one.mean() + two.mean()) / 2 / size), 5)


def render(model, folder, frame):
    if not os.path.isfile(os.path.join(folder, "view_3.png")):
        subprocess.run([BLENDER, "-b", "--python", os.path.join(PROJECT, "scripts/tools/render_views.py"), "--",
                        os.path.abspath(model), os.path.abspath(folder), "--frame", os.path.abspath(frame)],
                       check=True, stdout=subprocess.DEVNULL)
    return [Image.open(os.path.join(folder, f"view_{k}.png")).convert("RGBA") for k in range(4)]


BORDER = 5  # points above a limit that still count as borderline


def verdict_of(best, keep_min, part_min):
    """pass or redo, why, and the warnings, for the best candidate. The measured numbers decide. The AI saying it does
    not see the part makes it a redo only when the numbers are borderline (within BORDER of a limit): the measures can
    be fooled by a part of the right outline that is not there, but qwen3-vl:30b also said "not there" (1/10) of cora's
    tiara, which is in every render, while the numbers passed clearly (kept 98, part 82: Nov 19). The user's choice,
    Nov 22: keep the AI's veto, but only for borderline numbers. Otherwise it is a warning on the node."""
    why = [f"{key} {best[key]} is below {limit:g}" for key, limit in (("kept", keep_min), ("part", part_min)) if best[key] < limit]
    ai = best.get("ai") or {}
    warnings = []
    if ai.get("part_is_there") is False:
        said = f"the AI does not see the part ({ai.get('why') or 'no reason given'})"
        if best["kept"] < keep_min + BORDER or best["part"] < part_min + BORDER:
            why.append(said)
        else:
            warnings.append(said + f"; not a redo: the numbers pass clearly (kept {best['kept']}, part {best['part']})")
    if isinstance(ai.get("score"), (int, float)) and ai["score"] <= 3:
        warnings.append(f"AI score {ai['score']}: {ai.get('why') or ''}".strip())
    return ("redo" if why else "pass"), why, warnings


def main():
    parser = argparse.ArgumentParser(description="The local judge of an edit: is the part there, is the rest kept")
    parser.add_argument("--edit", required=True, help="the folder make_edit.py wrote")
    parser.add_argument("--original", required=True, help="the model before the edit")
    parser.add_argument("--candidates", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--step", choices=["shape", "details", "colours"], default="shape")
    parser.add_argument("--object", default="an object", help="what the object is (from its data sheet)")
    parser.add_argument("--part", default="a new part", help="what the new part is (from its data sheet)")
    parser.add_argument("--keep-min", type=float, default=85, help="kept below this: redo")
    parser.add_argument("--part-min", type=float, default=70, help="part below this: redo")
    parser.add_argument("--no-ai", action="store_true", help="only the measured scores")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    edit = json.load(open(os.path.join(args.edit, "edit.json")))
    frame = os.path.join(args.edit, "views.json")
    side = edit["side"]
    sides = 1 + len(edit.get("more_edit", [])) // 4  # the sides that have an AFTER picture and a mask
    name = lambda what, k: os.path.join(args.edit, f"{what}.png" if k == 0 else f"{what}_{k}.png")
    masks = [np.asarray(Image.open(name("mask", k)).convert("L")) > 127 for k in range(sides)]
    afters = [Image.open(name("after", k)) for k in range(sides)]
    originals = [Image.open(os.path.join(args.edit, f"view_{k}.png")).convert("RGBA") for k in range(4)]
    info, points = json.load(open(frame)), np.load(os.path.join(args.edit, "area.npy"))

    report = []
    for number, model in enumerate(args.candidates):
        folder = os.path.join(args.out, f"candidate{number + 1}")
        views = render(model, folder, frame)
        per_side = []
        for k in range(4):  # the edit's sides, then the sides without a picture: there only "kept" can be measured
            index = (side + k) % 4
            mask = masks[k] if k < sides else np.zeros(masks[0].shape, bool)
            one = measure_side(originals[index], views[index], mask, afters[k] if k < sides else None,
                               area_seen(points, info, index))
            if k >= sides:
                one.pop("part")
            if not edit.get("placed_in_3d"):
                for key in ("like", "like_shape", "like_colour"):
                    one.pop(key, None)
            per_side.append({"view": index, **one})
        parts = [s["part"] for s in per_side if "part" in s]
        colours = [s["colour"] for s in per_side if "colour" in s]
        entry = {"model": model, "sides": per_side, "kept": min(s["kept"] for s in per_side), "part": round(float(np.mean(parts))),
                 "colour": round(float(np.mean(colours))) if colours else None, "renders": folder,
                 "psnr": round(float(np.mean([s["psnr"] for s in per_side])), 1), "chamfer": chamfer_kept(args.original, model, points)}
        likes = [s["like_shape" if args.step != "colours" else "like"] for s in per_side if "like" in s]
        if likes:  # the wanted result is known from every side: how close to it, instead of "something changed"
            entry["like"] = round(float(np.mean(likes)))
            entry["part"] = entry["like"]
        entry["measured"] = min(entry["kept"], entry["part"], *([entry["colour"]] if args.step == "colours" and colours else []))
        report.append(entry)
        print(f"  {number + 1}: kept {entry['kept']}, part {entry['part']}, colour {entry['colour']}, PSNR outside {entry['psnr']} dB, "
              f"chamfer outside {entry['chamfer']}  ({os.path.basename(model)})", flush=True)

    if not args.no_ai:
        from judge import MODEL, ask, encoded
        for number, entry in enumerate(report):
            pictures = [name("after", 0), os.path.join(entry["renders"], f"view_{side}.png"),
                        os.path.join(entry["renders"], f"view_{(side + 1) % 4}.png")]
            grey = []
            for k, path in enumerate(pictures[1:]):  # the renders on the same grey as the collage
                grey.append(os.path.join(entry["renders"], f"ai_{k}.jpg"))
                Image.fromarray((on_grey(Image.open(path)) * 255).astype(np.uint8)).save(grey[-1], quality=92)
            try:
                answer, raw = ask([encoded(pictures[0], 768)] + [encoded(g, 768) for g in grey], STEP_NOTE[args.step] + QUESTION % (args.object, args.part))
            except Exception as e:  # noqa: BLE001 (Ollama not running: the measured scores still decide)
                answer, raw = {}, f"{type(e).__name__}: {e}"
            score = answer.get("score") if isinstance(answer.get("score"), (int, float)) else None
            entry["ai"] = {"model": MODEL, "score": score, "part_is_there": answer.get("part_is_there"),
                           "defects": answer.get("defects"), "why": answer.get("why"), "raw": raw[:600]}
            print(f"  {number + 1}: AI score {score}, defects {answer.get('defects')}", flush=True)

    # the best: the highest measured score; within 5 points the AI's score decides
    top = max(e["measured"] for e in report)
    close = [e for e in report if e["measured"] >= top - 5]
    best = max(close, key=lambda e: ((e.get("ai") or {}).get("score") or 0, e["measured"]))
    verdict, why, warnings = verdict_of(best, args.keep_min, args.part_min)
    result = {"step": args.step, "edit": args.edit, "original": args.original, "candidates": report, "best": best["model"],
              "verdict": verdict, "why": why, "warnings": warnings, "limits": {"kept": args.keep_min, "part": args.part_min}}
    json.dump(result, open(os.path.join(args.out, "judge.json"), "w"), indent=1)

    cell = 300  # wanted | original | every candidate, the edit's side first, then the next sides
    rows = [("wanted (AFTER pictures)", [name("after", k) for k in range(sides)]),
            ("original", [os.path.join(args.edit, f"view_{(side + k) % 4}.png") for k in range(4)])]
    for number, entry in enumerate(report):
        ai = entry.get("ai") or {}
        label = (f"{number + 1}: {os.path.basename(entry['model'])[-46:]}  kept {entry['kept']}  part {entry['part']}"
                 f"  colour {entry['colour']}  AI {ai.get('score')}" + ("   <- best" if entry is best else ""))
        rows.append((label, [os.path.join(entry["renders"], f"view_{(side + k) % 4}.png") for k in range(4)]))
    sheet = Image.new("RGB", (4 * cell, len(rows) * (cell + 24) + 24), (40, 40, 44))
    draw = ImageDraw.Draw(sheet)
    for r, (label, pictures) in enumerate(rows):
        draw.text((6, r * (cell + 24) + 6), label, fill=(255, 255, 255))
        for c, path in enumerate(pictures):
            image = Image.fromarray((on_grey(Image.open(path)) * 255).astype(np.uint8))
            image.thumbnail((cell, cell))
            sheet.paste(image, (c * cell, r * (cell + 24) + 24))
    draw.text((6, len(rows) * (cell + 24) + 6), f"{args.step}: {verdict}" + (f" ({'; '.join(why)})" if why else ""),
              fill=(120, 220, 140) if verdict == "pass" else (240, 150, 90))
    sheet.save(os.path.join(args.out, "compare.jpg"), quality=90)
    print(f"best: {os.path.basename(best['model'])}, {verdict} {why}", flush=True)
    print(f"Result: {os.path.join(args.out, 'judge.json')}", flush=True)


if __name__ == "__main__":
    main()
