"""The judge BEFORE the inpainting: are the edit's pictures and its 3D area fit to be sent? An inpainting takes 20 to
40 minutes on the uni server; what is wrong in its input can be seen in a minute on this PC.

The toaster robot's toast -> book (Nov 15) went to the server twice with faults nobody had looked for: only the front
had a picture, and the 3D area was a column 18 voxels deep in front of the robot for a book 3 voxels thick (its middle
was put on the nearest surface, not on the part). The result: a thin torn slab and loose pieces.

    measured, per side that has a picture (the 2D mask next to the 3D area seen from that side):
      outside   the share of the new part's picture that lies outside the 3D area: it cannot be built there
      room      how many times bigger the 3D area is than the part in the picture: loose pieces grow in empty room
      middle    how far the area's middle is from the part's, in part widths
    over all sides: how many have a picture; with one, how deep the area is (nothing says where in it the part is)
    seen: the local vision AI (Qwen3.6-27B, Ollama) gets BEFORE and AFTER of every side: is it the part that was
          asked for, the same one on every side, held or attached where it should be, at a sensible size?

Verdict: pass, or redo with what is wrong and what to do about it. cloud.py edit asks for it before it sends.

    python scripts/pipeline/edit_check.py outputs/sad_toaster/work/edit_3d --part "a closed dark red book held in the hand"
"""
import _paths  # noqa: F401  (the scripts of the other folders)

import argparse
import json
import os

import numpy as np
from PIL import Image, ImageDraw

from edit_judge import area_seen

SIDES = ["the main side", "the next side (a quarter turn)", "the opposite side", "the last side (three quarter turns)"]
QUESTION = """These pictures show a planned change of a 3D model of %s, from %d side(s). Per side: BEFORE on the left, the
wanted AFTER on the right. The change: %s.
Judge only the plan, strictly:
 - is the new part in the AFTER pictures what was asked for, complete, at a sensible size for the object?
 - is it the same part, at the same place, on every side that shows it?
 - is it held or attached where it should be (not floating, not cutting through the object)?
 - is the old part it replaces gone on every side?
Answer only with JSON, no other text:
{"score": 1-10, "ready": true or false, "defects": ["at most 4 short points"]}"""


def measure(folder):
    edit = json.load(open(os.path.join(folder, "edit.json")))
    info, points = json.load(open(os.path.join(folder, "views.json"))), np.load(os.path.join(folder, "area.npy"))
    side, sides = edit["side"], 1 + len(edit.get("more_edit", [])) // 4
    name = lambda what, k: os.path.join(folder, f"{what}.png" if k == 0 else f"{what}_{k}.png")
    voxel_pixels = info["span"] / 1.4 / 64 / (info["span"] / info["size"])
    facts = []
    view_of = edit.get("view_of") or [(side + k) % 4 for k in range(sides)]
    for k in range(sides):
        mask = np.asarray(Image.open(name("mask", k)).convert("L")) > 127
        area = area_seen(points, info, view_of[k], voxels=1.0)
        one = {"side": k, "part_share": round(float(mask.mean()), 4)}
        if mask.sum() > 100:
            centre = lambda m: np.array(np.nonzero(m)).mean(1)
            one.update(outside=round(float((mask & ~area).sum() / mask.sum()), 3),
                       room=round(float(area.sum() / mask.sum()), 2),
                       middle=round(float(np.linalg.norm(centre(area) - centre(mask)) / np.sqrt(mask.sum())), 2))
        facts.append(one)
    to_gltf = lambda p: np.array([p[0], p[2], -p[1]])
    toward = to_gltf(info["views"][side]["toward"])
    right = to_gltf(info["views"][side]["right"])
    voxel = info["span"] / 1.4 / 64
    cut_off = None
    if "cube" in edit:  # TRELLIS.2's cube ends at the model's box: what reaches past it is cut off
        low, high = (np.array(c) for c in edit["cube"])
        cut_off = round(float(np.any((points < low) | (points > high), 1).mean()), 3)
        voxel = float((high - low).max() / 64)
    return {"sides": facts, "voxels": int(len(points) / 8), "voxel_pixels": round(voxel_pixels, 1), "cut_off": cut_off,
            "part_voxels": edit.get("part_voxels"),
            "deep": round(float(np.ptp(points @ toward) / voxel), 1), "wide": round(float(np.ptp(points @ right) / voxel), 1),
            "high": round(float(np.ptp(points[:, 1]) / voxel), 1)}


def findings(measured, grows=False):
    """What is wrong, as (how bad 1..3, what, what to do). 3 = do not send."""
    found = []
    shown = [f for f in measured["sides"] if "room" in f]
    roomy = [f for f in shown if f["room"] > 3]
    for f in shown:
        where = SIDES[f["side"]] if f["side"] < len(SIDES) else f"side {f['side'] + 1} (in between)"
        if f["outside"] > 0.3:
            found.append((3, f"On {where}, {f['outside']:.0%} of the new part's picture lies outside the 3D area: it cannot be built there.",
                          "Make the area again from the same pictures (make_edit.py or place_part.py)."))
        if f["room"] > 3:
            # from one side the model itself may hide most of the part (a book in front of the chest, seen from behind):
            # do not send only if the main side says so, or half of the sides
            bad = 3 if f["side"] == 0 or len(roomy) >= len(shown) / 2 else 1
            found.append((bad, f"On {where}, the 3D area is {f['room']:.1f} times as big as the part in the picture"
                          + (": loose pieces grow in the empty room." if bad == 3 else " (the model may hide the part from there)."),
                          "Give the part's own 3D model (place_part.py): the area is then the part and nothing else."))
        elif f["middle"] > 0.35:
            found.append((2, f"On {where}, the middle of the 3D area is {f['middle']:.1f} part widths away from the part.",
                          "Look at area_map.jpg: the area should lie on the part."))
    if len(shown) <= 1:
        # grows: a part from words that grows out of the model (a crown on a head) has only the front by design, and its
        # shape is made from noise there (the crown and the wheel, October); for it this is a warning, not a stop
        # (Nov 19: the rule stopped every words-only flow of the web app before its shape step)
        found.append((2 if grows or measured["deep"] <= 8 else 3,
                      f"Only one side has a picture of the new part. The 3D area is {measured['deep']:.0f} voxels deep "
                      f"({measured['wide']:.0f} wide, {measured['high']:.0f} high), and nothing says where in it the part is or how thick: "
                      "its depth, sides and back would be guessed.",
                      "Make the part's own 3D model (generate.py from its picture) and place it: place_part.py gives all four sides."))
    if not shown:
        found.append((3, "No side shows a change.", "Make the edit again."))
    if (measured.get("cut_off") or 0) > 0.02:
        found.append((3, f"{measured['cut_off']:.0%} of the 3D area lies outside the model's box: TRELLIS.2's cube ends there, the part would be cut off.",
                      "Place the part with place_part.py (its pegs make the box big enough), or make the part smaller."))
    thin = min(measured.get("part_voxels") or [99])
    if thin < 3:
        found.append((3, f"The new part is only {thin:.1f} voxels thick: too thin to be built at all.",
                      "Zoom in (place_part.py --zoom 0.25 sends only a box around the part) and let place_part.py make it "
                      "thicker (--min-voxels 5)."))
    elif thin < 4.5:  # Nov 19: a book 3.6 voxels thick came out as a frame with hollow, speckled covers; 5 came out whole
        found.append((2, f"The new part is only {thin:.1f} voxels thick: its flat sides may come out hollow.",
                      "place_part.py --min-voxels 5 makes a flat part that thick (the flow's place node does it)."))
    return sorted(found, key=lambda f: -f[0])


def sheet(folder, sides):
    """BEFORE | AFTER of every side, in one picture (also what the local AI sees)."""
    name = lambda what, k: os.path.join(folder, f"{what}.png" if k == 0 else f"{what}_{k}.png")
    cell = 384
    picture = Image.new("RGB", (sides * (2 * cell + 16), cell + 24), (40, 40, 44))
    draw = ImageDraw.Draw(picture)
    for k in range(sides):
        x = k * (2 * cell + 16)
        for i, what in enumerate(("before", "after")):
            picture.paste(Image.open(name(what, k)).convert("RGB").resize((cell, cell), Image.LANCZOS), (x + i * cell, 24))
        draw.text((x + 6, 6), f"side {k + 1}: before | after", fill=(235, 235, 235))
    path = os.path.join(folder, "check.jpg")
    picture.save(path, quality=92)
    return path


def check(folder, object_="an object", part="a new part", ai=True, grows=False):
    measured = measure(folder)
    found = findings(measured, grows)
    sides = len(measured["sides"])
    seen = {}
    if ai:
        from judge import MODEL, ask, encoded
        try:
            answer, raw = ask([encoded(sheet(folder, sides), 1536)], QUESTION % (object_, sides, part))
        except Exception as e:  # noqa: BLE001 (Ollama not running: what is measured still decides)
            answer, raw = {}, f"{type(e).__name__}: {e}"
        seen = {"model": MODEL, "score": answer.get("score"), "ready": answer.get("ready"),
                "defects": answer.get("defects"), "raw": raw[:600]}
        if isinstance(seen["score"], (int, float)) and seen["score"] <= 4:
            found.append((3, f"The local AI gives the plan {seen['score']} of 10: {'; '.join(seen['defects'] or [])}",
                          "Make the AFTER pictures again (other words, another --prompt-seed, or the part placed in 3D)."))
            found.sort(key=lambda f: -f[0])
    verdict = "redo" if any(f[0] == 3 for f in found) else "pass"
    result = {"edit": folder, "verdict": verdict, "measured": measured, "seen": seen,
              "findings": [{"how_bad": f[0], "what": f[1], "do": f[2]} for f in found]}
    json.dump(result, open(os.path.join(folder, "check.json"), "w"), indent=1)
    return result


def main():
    parser = argparse.ArgumentParser(description="The judge before the inpainting: the edit's pictures and its 3D area")
    parser.add_argument("edit", help="the folder make_edit.py or place_part.py wrote")
    parser.add_argument("--object", default="an object")
    parser.add_argument("--part", default="a new part", help="what the change is, in words")
    parser.add_argument("--no-ai", action="store_true", help="only what is measured")
    parser.add_argument("--grows", action="store_true", help="a part from words that grows out of the model: one side "
                        "is expected (a warning, not a redo)")
    parser.add_argument("--out", help="also write check.json (and check.jpg) here, and passed.json only on pass: a flow "
                        "node's folder, so a plan judged redo is judged again next time and not taken as done")
    args = parser.parse_args()
    result = check(args.edit, args.object, args.part, not args.no_ai, args.grows)
    if args.out:
        import shutil
        os.makedirs(args.out, exist_ok=True)
        for name in ("check.json", "check.jpg"):
            if os.path.isfile(os.path.join(args.edit, name)):
                shutil.copy(os.path.join(args.edit, name), os.path.join(args.out, name))
        passed = os.path.join(args.out, "passed.json")
        if result["verdict"] == "pass":
            json.dump({"verdict": "pass"}, open(passed, "w"))
        elif os.path.exists(passed):
            os.remove(passed)
    m = result["measured"]
    print(f"the 3D area: {m['voxels']} voxels, {m['wide']:.0f} wide x {m['high']:.0f} high x {m['deep']:.0f} deep"
          + (f"; the new part is {' x '.join(f'{v:.0f}' for v in m['part_voxels'])} voxels" if m.get("part_voxels") else ""), flush=True)
    for f in m["sides"]:
        print(f"  side {f['side'] + 1}: " + (f"the part is {100 * f['part_share']:.1f} % of the picture, {f['outside']:.0%} of it outside "
              f"the area, the area {f['room']:.1f} times its size, middles {f['middle']:.2f} part widths apart" if "room" in f
              else "no change to see"), flush=True)
    if result["seen"]:
        print(f"the local AI: {result['seen'].get('score')} of 10, defects {result['seen'].get('defects')}", flush=True)
    for f in result["findings"]:
        print(f"  [{f['how_bad']}] {f['what']}\n      -> {f['do']}", flush=True)
    print(f"verdict: {result['verdict']}", flush=True)
    return 0 if result["verdict"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
