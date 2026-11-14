"""The picture of a change (the add-on's "1. Make picture") judged before 25 minutes of regenerating, and made better.

    judge    what can be measured on the pictures of every side, turned into plain advice; then the local vision AI
             (Qwen3.6-27B, Ollama, as judge.py) looks at the before / after sheet and says what it thinks
    improve  the local picture AI (Qwen-Image-Edit-2511 through ComfyUI, as picture_from_prompt.py) reworks the AFTER
             picture of every side as the words say ("join the head to the neck with a metal collar"); outside the
             change the picture stays the exact BEFORE, and the 2D mask becomes what differs from BEFORE
    enhance  both in turn, by itself: judge, rework with the judge's own sentence (and the user's words), judge again; a
             round stays only if the picture got better, and it ends when a round didn't help (as the grid loop)
    restore  the pictures as they were before the first improve
    parts    before all that: the new part's own pictures made ready for the add-on, every side cut out on transparent.
             The add-on only pastes the part in (no AI in Blender): a picture with a studio background came in with a
             dark disc around it, a folder of More sides with a wireframe drawing in it gave that as the side view
             (the bee robot's new head, Nov 14). A video of the part turning once gives its four sides a quarter turn
             apart (front first); a picture (+ a folder of up to 3 more sides) is cut out by the background AI
             TRELLIS.2 uses; a picture that is transparent already is only cropped. Writes <out>/side0.png ... and
             part.json; run again with the same files, it answers at once.

    source activate.sh
    python scripts/inpainting/picture_check.py judge outputs/blender/geometry_0_picture_seed1 --what "a new robot head"
    python scripts/inpainting/picture_check.py improve outputs/blender/geometry_0_picture_seed1 --prompt "..."
    python scripts/inpainting/picture_check.py restore outputs/blender/geometry_0_picture_seed1
    python scripts/inpainting/picture_check.py parts "pictures/robot head 2/helmet.mp4" --out outputs/blender/part
    python scripts/inpainting/picture_check.py parts crown.jpg --sides pictures/crown_sides --out outputs/blender/part

<stem> is the picture's path without ".png". Its sides are <stem>.png and <stem>_side1.png ..., each with _before.png,
_mask.png and, from the add-on, _old.png (where the part that goes shows). judge writes <stem>_judge.json.

What is measured (per side, on the object's pixels):
    left over    the share of the old part that still shows beside the new one
    on kept      the share of the new part that lies over surface that stays
    height       the new part's height as a share of the whole model's
    touches      whether the new part touches the rest of the model (or floats)
"""
import _paths  # noqa: F401  (the scripts of the other folders)

import argparse
import json
import os
import shutil

import numpy as np
from PIL import Image
from scipy import ndimage

GREY = np.array([190, 190, 192])  # the add-on's background
SIDE_NAMES = ("side 1 (your view)", "side 2", "side 3", "side 4")  # as the columns of the before / after sheet
ENDS = (".png", "_mask.png")  # what improve changes (and restore brings back)

QUESTION = """This sheet shows a planned change of a 3D model. Top row: the model BEFORE. Bottom row: the model AFTER,
with a new part pasted onto it. Every column is one side of the model (the same side above and below).
The new part should be: {what}.
Measured on the pictures: {facts}

Judge the AFTER row as a plan for a 3D model: does the new part have a fitting size, does it sit in the right place
and join the rest of the model, does it face the same way as the body on every side, is anything of the old part left
that should be gone, does its style fit?
Answer only with JSON, no other text:
{{"score": a number from 1 to 10 (10 = a convincing plan),
 "good": ["what is right, at most 3 short points"],
 "problems": ["what is wrong, at most 4 short points, the worst first"],
 "better": ["what the user should do to make it better, at most 3 short points"],
 "prompt": "one sentence telling a picture-editing AI how to rework the AFTER pictures so the new part fits better"}}"""

IMPROVE = ("Image 1 shows a 3D model on a plain grey background; a new part was pasted onto it. {prompt}{reference} "
           "The new part is joined to the model where the old part was: it goes into the opening there, as a neck goes "
           "into a collar, it does not lie on top or float, and nothing of the old part is left. "
           "Keep the camera angle, the lighting, the plain grey background and every other part of the model exactly as they are.")


PICTURE_TYPES = (".png", ".jpg", ".jpeg", ".webp")
VIDEO_TYPES = (".mp4", ".mov", ".webm", ".avi", ".mkv", ".m4v")


def stamp(path):
    return [os.path.abspath(path), os.path.getmtime(path), os.path.getsize(path)]


def parts(args):
    """The new part's pictures made ready for the add-on's "1. Make picture": every side cut out on transparent."""
    args.reference = args.stem
    video = args.reference.lower().endswith(VIDEO_TYPES)
    sources = [args.reference]
    if args.sides and not video:
        sources += [os.path.join(args.sides, n) for n in sorted(os.listdir(args.sides)) if n.lower().endswith(PICTURE_TYPES)][:3]
    stamps = [stamp(s) for s in sources]
    record = os.path.join(args.out, "part.json")
    if os.path.isfile(record):
        old = json.load(open(record))
        if old.get("sources") == stamps and all(os.path.isfile(p) for p in old["pictures"]):
            print(f"Result: {record} (already there)", flush=True)
            return
    os.makedirs(args.out, exist_ok=True)

    import make_edit
    if video:
        from data_sheet import video_views
        sources = video_views(args.reference, args.out, "frame", 4)  # 0, 90, 180, 270 degrees of its turn
    remover, pictures = None, []
    for k, source in enumerate(sources):
        picture = Image.open(source)
        alpha = np.asarray(picture.convert("RGBA"))[..., 3]
        if alpha.min() < 128:  # transparent already: only cropped
            ys, xs = np.nonzero(alpha > 128)
            cut, how = picture.convert("RGBA").crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)), "its own transparency"
        else:
            remover = remover or make_edit.background_remover()
            cut, how = make_edit.cut_out(remover, source), "cut out by the background AI"
        pictures.append(os.path.join(os.path.abspath(args.out), f"side{k}.png"))
        cut.save(pictures[-1])
        print(f"  side {k}: {os.path.basename(source)}, {cut.width} x {cut.height}, {how}", flush=True)
    json.dump({"pictures": pictures, "sources": stamps, "video": video}, open(record, "w"), indent=1)
    print(f"Result: {record}", flush=True)


def sides_of(stem):
    return [stem] + [f"{stem}_side{k}" for k in range(1, 4) if os.path.isfile(f"{stem}_side{k}.png")]


def picture(path, size=None):
    image = Image.open(path).convert("RGB")
    return np.asarray(image if size is None else image.resize((size, size), Image.LANCZOS), int)


def measure(stem):
    """What can be measured on one side's pictures."""
    before, after = picture(stem + "_before.png"), picture(stem + ".png")
    mask = picture(stem + "_mask.png")[..., 0] > 127
    model = np.abs(before - GREY).sum(-1) > 24  # the model's pixels before
    whole = model | mask
    rows = np.flatnonzero(whole.any(1))
    part_rows = np.flatnonzero(mask.any(1))
    facts = {"part_share_of_height": round(float(np.ptp(part_rows) + 1) / max(1, np.ptp(rows) + 1), 3) if len(part_rows) else 0.0,
             "at_border": bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())}
    old = None
    if os.path.isfile(stem + "_old.png"):
        old = picture(stem + "_old.png")[..., 0] > 127
        if old.any():
            facts["left_over"] = round(float((old & ~mask).sum()) / float(old.sum()), 3)
    stays = model & ~mask & ~(old if old is not None else np.zeros_like(mask))  # what the new part should join
    facts["on_kept"] = round(float((mask & model & ~(old if old is not None else mask)).sum()) / max(1.0, float(mask.sum())), 3) \
        if old is not None else None
    facts["touches"] = bool((ndimage.binary_dilation(mask, iterations=8) & stays).any()) if stays.any() else True
    facts["changed_outside"] = round(float(((np.abs(after - before).sum(-1) > 40) & ~ndimage.binary_dilation(mask, iterations=6)).mean()), 4)
    return facts


def findings(measured):
    """The measured facts as plain sentences: (how bad 0..3, what is wrong, what to do about it)."""
    found = []
    for name, facts in zip(SIDE_NAMES, measured):
        if facts.get("left_over", 0) > 0.12:
            found.append((2 if facts["left_over"] > 0.3 else 1,
                          f"On {name}, {facts['left_over']:.0%} of the old part still shows beside the new one.",
                          "Make the new part cover it: a bigger space box (the part is scaled to the box), or say in "
                          "'Make it better' what should be there instead."))
        if (facts.get("on_kept") or 0) > 0.12:
            found.append((2 if facts["on_kept"] > 0.3 else 1,
                          f"On {name}, {facts['on_kept']:.0%} of the new part lies over surface that stays.",
                          "Press 'Fit to colours' so the box ends where the red ends, or paint that surface yellow (E) "
                          "so the AI may change it."))
        if not facts["touches"]:
            found.append((3, f"On {name}, the new part doesn't touch the rest of the model: it would float.",
                          "Move the bottom of the space box down to where the part should sit."))
        if facts["at_border"]:
            found.append((2, f"On {name}, the new part is cut off at the picture's border.", "Make the space box smaller."))
    heights = [f["part_share_of_height"] for f in measured]
    if max(heights) > 0.6:
        found.append((1, f"The new part is {max(heights):.0%} of the model's height.",
                      "If that is too big: a smaller space box makes it smaller."))
    if len(measured) == 1:
        found.append((1, "Only one side of the new part is given: its back and sides will be guessed.",
                      "Use a video of the part turning once as the Reference, or a folder of More sides."))
    return sorted(found, key=lambda f: -f[0])


def judge(args):
    judge_once(args.stem, args.what, args.no_ai)
    print(f"Result: {args.stem}_judge.json", flush=True)


def total(result):
    """One number to compare two judgings: what was measured, and the local AI's score when it answered."""
    return result["score_measured"] + (result["ai"]["score"] if result.get("ai") else result["score_measured"])


def judge_once(stem, what, no_ai=False):
    args = argparse.Namespace(stem=stem, what=what, no_ai=no_ai)
    sides = sides_of(args.stem)
    measured = [measure(side) for side in sides]
    found = findings(measured)
    score = max(1, 10 - sum(bad for bad, _, _ in found))
    result = {"stem": args.stem, "sides": len(sides), "measured": measured, "score_measured": score,
              "findings": [{"bad": bad, "what": what, "do": do} for bad, what, do in found], "ai": None, "ai_error": None}
    record = args.stem + "_judge.json"
    json.dump(result, open(record, "w"), indent=1)  # the measured part is there at once
    print(f"measured: {score}/10, {len(found)} finding(s)", flush=True)
    for bad, what, do in found:
        print(f"  [{bad}] {what} -> {do}", flush=True)
    if not args.no_ai:
        sheet = args.stem + "_preview.jpg"
        if not os.path.isfile(sheet):  # as the add-on's: the top row before, the bottom row after
            cell = 512
            image = Image.new("RGB", (cell * len(sides), 2 * cell))
            for column, side in enumerate(sides):
                for row, end in enumerate(("_before.png", ".png")):
                    image.paste(Image.open(side + end).convert("RGB").resize((cell, cell), Image.LANCZOS), (column * cell, row * cell))
            image.save(sheet, quality=90)
        try:
            from judge import MODEL, ask, encoded
            facts = "; ".join(f"{name}: " + ", ".join(f"{k} {v}" for k, v in m.items() if v is not None)
                              for name, m in zip(SIDE_NAMES, measured))
            answer, _ = ask([encoded(sheet, 1536)], QUESTION.format(what=args.what or "not said", facts=facts))
            lines = lambda key, most: [str(x)[:160] for x in (answer.get(key) or [])][:most]
            result["ai"] = {"model": MODEL, "score": max(1, min(10, int(round(float(answer.get("score", 0)))))),
                            "good": lines("good", 3), "problems": lines("problems", 4), "better": lines("better", 3),
                            "prompt": str(answer.get("prompt", ""))[:300]}
            print(f"the local AI: {result['ai']['score']}/10, {result['ai']['problems']}", flush=True)
        except Exception as e:  # noqa: BLE001 (Ollama not running or no room on the GPU: the measured part stays)
            result["ai_error"] = f"{type(e).__name__}: {e}"[:200]
            print(f"the local AI didn't answer: {result['ai_error']}", flush=True)
        json.dump(result, open(record, "w"), indent=1)
    return result


def enhance(args):
    """Judge, rework, judge again; a round stays only if the picture got better."""
    best = judge_once(args.stem, args.what)
    rounds = []
    for number in range(1, args.rounds + 1):
        idea = (best.get("ai") or {}).get("prompt", "")
        fixes = " ".join(f["what"] for f in best["findings"] if f["bad"] >= 2)
        words = " ".join(part for part in (args.prompt.strip(), idea, f"Fix this: {fixes}" if fixes else "") if part)
        if not words:
            print("nothing to make better: no finding and no idea from the local AI", flush=True)
            break
        print(f"round {number}: {words[:200]}", flush=True)
        sides = sides_of(args.stem)
        for side in sides:  # this round can be taken back
            for end in ENDS:
                shutil.copy(side + end, side + "_round" + end)
        improve(argparse.Namespace(stem=args.stem, prompt=words, reference=args.reference, seed=args.seed + 10 * number))
        new = judge_once(args.stem, args.what)
        better = total(new) > total(best)
        rounds.append({"round": number, "words": words, "before": total(best), "after": total(new), "kept": better})
        print(f"round {number}: {total(best)} -> {total(new)}: {'kept' if better else 'not better, taken back'}", flush=True)
        for side in sides:
            for end in ENDS:
                if not better:
                    shutil.copy(side + "_round" + end, side + end)
                os.remove(side + "_round" + end)
        if not better:
            break
        best = new
    best["rounds"] = rounds
    json.dump(best, open(args.stem + "_judge.json", "w"), indent=1)
    print(f"Result: {args.stem}.png ({sum(r['kept'] for r in rounds)} round(s) kept)", flush=True)


def improve(args):
    from picture_from_prompt import comfy_edit, start_comfyui
    sides = sides_of(args.stem)
    for side in sides:  # the first pictures are kept, for restore
        for end in ENDS:
            if not os.path.isfile(side + "_first" + end):
                shutil.copy(side + end, side + "_first" + end)
    reference = Image.open(args.reference).convert("RGB") if args.reference else None
    started = start_comfyui(lambda text: print(text, flush=True))  # once for all sides
    try:
        for k, side in enumerate(sides):
            print(f"improving side {k + 1}/{len(sides)}", flush=True)
            before = picture(side + "_before.png")
            size = before.shape[0]
            after = Image.open(side + ".png").convert("RGB")
            old_mask = picture(side + "_mask.png")[..., 0] > 127
            old = picture(side + "_old.png")[..., 0] > 127 if os.path.isfile(side + "_old.png") else np.zeros_like(old_mask)
            edited = np.asarray(comfy_edit(after, args.prompt, args.seed + k, reference, lambda text: None, template=IMPROVE)
                                .resize((size, size), Image.LANCZOS), int)
            # only around the change: the new part, the old part and a little room; the rest stays the exact BEFORE
            region = ndimage.binary_dilation(old_mask | old, iterations=max(4, size // 16))
            soft = ndimage.gaussian_filter(region.astype(np.float32), max(2, size // 200))[..., None]
            result = (edited * soft + before * (1 - soft)).astype(np.uint8)
            differs = (np.abs(result.astype(int) - before).sum(-1) > 40) & region
            differs = ndimage.binary_fill_holes(ndimage.binary_closing(differs, iterations=3))
            labels, count = ndimage.label(differs)
            if count:  # the big pieces of change only (specks of noise are no part)
                sizes = ndimage.sum(differs, labels, range(1, count + 1))
                differs = np.isin(labels, 1 + np.flatnonzero(sizes >= 0.05 * sizes.max()))
            if differs.sum() < 50:
                print("  nothing changed on this side: kept as it was", flush=True)
                continue
            Image.fromarray(result).save(side + ".png")
            Image.fromarray((ndimage.binary_dilation(differs, iterations=4) * 255).astype(np.uint8)).convert("RGB").save(side + "_mask.png")
    finally:
        if started is not None:
            started.terminate()
            started.wait(timeout=60)
    print(f"Result: {args.stem}.png", flush=True)


def restore(args):
    brought = 0
    for side in sides_of(args.stem):
        for end in ENDS:
            if os.path.isfile(side + "_first" + end):
                shutil.copy(side + "_first" + end, side + end)
                brought += 1
    print(f"Result: {args.stem}.png ({brought} picture(s) as they were first)", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Judge the picture of a change, or make it better with the picture AI")
    parser.add_argument("what_to_do", choices=["judge", "improve", "enhance", "restore", "parts"])
    parser.add_argument("stem", help="the picture's path without .png (parts: a picture or a video of the new part)")
    parser.add_argument("--sides", default="", help="parts: a folder with up to 3 more pictures, a quarter turn further each")
    parser.add_argument("--out", help="parts: the folder for the cut-out sides")
    parser.add_argument("--what", default="", help="judge: what the new part should be, in the user's words")
    parser.add_argument("--no-ai", action="store_true", help="judge: only what is measured")
    parser.add_argument("--prompt", default="", help="improve: how the pictures should be reworked")
    parser.add_argument("--reference", help="improve: a picture of the new part, for the picture AI to look at")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=2, help="enhance: at most this many rounds")
    args = parser.parse_args()
    if args.what_to_do == "parts":
        if not args.out:
            raise SystemExit("parts needs --out")
        return parts(args)
    if args.stem.lower().endswith(".png"):
        args.stem = args.stem[:-4]
    if not os.path.isfile(args.stem + ".png"):
        raise SystemExit(f"no picture {args.stem}.png")
    if args.what_to_do == "improve" and not args.prompt.strip():
        raise SystemExit("improve needs --prompt: how the picture should be better")
    {"judge": judge, "improve": improve, "enhance": enhance, "restore": restore}[args.what_to_do](args)


if __name__ == "__main__":
    main()
