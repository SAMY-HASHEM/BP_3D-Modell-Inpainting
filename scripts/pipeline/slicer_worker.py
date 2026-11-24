"""The Slicer's worker: does what the page asks for, one job at a time on one GPU.

Every second it looks into outputs/slicer/*/:
- a plan_request.json: Qwen3-VL looks at the pictures and the parts are planned (slicer_plan.py);
- a message you typed that has no answer yet: Qwen3-VL answers it in the chat. It sees your pictures, the plan, the
  step you are on and the chat so far, and it can change the current step's prompt when you ask for that.
- a picture asked for in the chat, or a step's Make it (parts/<id>/request.json): Qwen-Image-Edit makes it and
  BiRefNet cuts it out (slicer_pictures.py); Qwen3-VL leaves the GPU meanwhile. The 3D models come next.

One model on the GPU at a time: Qwen3-VL (about 17 GB) for the chat, or ComfyUI's Qwen-Image-Edit (about 32 GB)
for a picture; each one is stopped before the other loads. Whatever is loaded is let go after 10 quiet minutes.

    python scripts/pipeline/slicer_worker.py          # keeps running; Ctrl+C stops it
"""
import gc
import json
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(os.path.dirname(HERE))
ROOT = os.path.join(PROJECT, "outputs", "slicer")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "generation"))
from slicer_page import add_made, answer, chat_state, slicer_says, unanswered  # noqa: E402
import slicer_plan  # noqa: E402  (sets the model folder, offline)

IDLE = 600  # seconds without work before Qwen3-VL leaves the GPU
_model = None
_last = 0.0

CHAT = """You are the Slicer, the assistant inside a local app (no cloud) that turns a character picture into a 3D
model in layers: first the whole character in an A-pose, then its base body (the same body without outfit and
accessories), then every piece of clothing, every accessory and every prop alone, to be put on the body afterwards. For every part the app makes a coloured picture and the same as grey clay (2048 x
2048, flat #808080 grey background), then a 3D model from them, shown on the right of the page.

Pictures are named by number: "#3" or "picture 3" is picture 3. You see these (in order): {pictures}.
The A-pose picture is {apose}: every base body and every piece alone starts from it. The character: {character}.
The parts: {parts}.
The step now: {step}
Its prompt for the image model now: {prompt}
Waiting for the image model: {waiting}
What works: planning the parts, this chat, changing prompts, and MAKING PICTURES: you can edit any of the user's
pictures with the local image model (about a minute). Not connected yet: the 3D models (say so honestly).

The chat so far:
{history}

Answer the user's last message. Be short and concrete (at most 5 sentences), in the user's language, plain text.
If the user asks for a picture (to make, change, redo or show one: a pose, only the character, no background, a
part alone, grey clay, from the back ...), fill "picture": "from" is the number of the picture to start from (the
newest fitting one if not said), "instruction" is ONE precise English instruction for an image-edit model that
describes the whole result: who it is, the exact pose (an A-pose: standing upright facing the camera, both arms
straight down and held away from the body at 45 degrees, palms toward the thighs, legs straight and slightly apart,
feet flat), what to remove, what to keep exactly the same (face, hair, every piece of the outfit), full body
visible and centred, plain flat grey background (never "transparent" or "no background": the background is cut out
afterwards). In "reply" then say briefly what you will make. Leave "picture" empty otherwise.
A "base body" / "base model" / "naked" / "without clothes" / "without the outfit" request is normal here and never
refused: it is the base body of 3D character work. Start it from the A-pose picture; its instruction is exactly:
"{base}". Explain in "reply" that the base body is a matte grey shop mannequin with exactly the character's
body (no anatomy, nothing sexual), the usual base for 3D clothes.
A piece "alone" (a jacket, boots, ears, a weapon) starts from the A-pose picture too and stays where it is, so all
layers fit together; its instruction is exactly: "Keep only <the piece> {alone}." (never the word "mannequin"
alone: the model then draws a whole mannequin).
Only if the user clearly asks to change the prompt of the step now, write the whole new prompt in "new_prompt", in
English, keeping the locked part ("Locked: ...") unchanged at its end; otherwise "new_prompt" stays "".
In "picture", "only" is the short name of the one piece asked for alone ("bodysuit", "bunny ears", "boots"), or
"" for a whole character or the base body: then everything else is cut away from the result.
Answer with JSON only: {{"reply": "...", "new_prompt": "", "picture": {{"from": 0, "instruction": "", "only": ""}}}}"""


def model():
    """Qwen3-VL on the GPU. Only one model at a time: the picture model (ComfyUI, about 32 GB once it has made a
    picture: it keeps its weights) is stopped first, or both together overflow the GPU (45 of 44 GB, Nov 24)."""
    global _model
    if _model is None:
        if "slicer_pictures" in sys.modules:
            sys.modules["slicer_pictures"].stop()
        from organise_pictures import load
        _model = load("Qwen/Qwen3-VL-8B-Instruct", False)
    return _model


def let_go(quiet=True):
    """Qwen3-VL off the GPU; when quiet (no work for a while) also ComfyUI and BiRefNet."""
    global _model
    if quiet and "slicer_pictures" in sys.modules:
        sys.modules["slicer_pictures"].stop()
    if _model is not None:
        import torch
        _model = None
        gc.collect()
        torch.cuda.empty_cache()
        print(time.strftime("%H:%M:%S"), "Qwen3-VL let go", "(quiet)" if quiet else "(for a picture)", flush=True)


def plan(name):
    folder = os.path.join(ROOT, name)
    note = json.load(open(os.path.join(folder, "plan_request.json"))).get("note", "")
    print(time.strftime("%H:%M:%S"), "planning", name, flush=True)
    try:
        slicer_plan.make_plan(folder, note=note, model=model())
    except BaseException as e:  # make_plan's SystemExit included
        slicer_says(name, f"I couldn't plan the parts: {str(e)[:300]}\nTell me in the chat what the picture shows, "
                          "or send another one.")
    finally:
        if os.path.isfile(os.path.join(folder, "plan_request.json")):
            os.remove(os.path.join(folder, "plan_request.json"))


def reply(name, upto):
    from organise_pictures import ask, shrink
    from describe_colours import on_grey
    from PIL import Image
    st = chat_state(name)
    raw = json.load(open(os.path.join(ROOT, name, "slicer.json")))
    sources = raw.get("sources") or ([raw["source"]] if raw.get("source") else [])
    parts = raw.get("parts", [])
    step = parts[st["step"]] if st["step"] < len(parts) else None
    images, labels = [], []
    steps = {v: k for k, v in (raw.get("step_pictures") or {}).items()}
    made = set(raw.get("made") or [])
    show = sorted(set(range(min(len(sources), 2))) | set(range(max(0, len(sources) - 3), len(sources))))
    for i in show:  # the first two and the newest three: the model sees at most five
        images.append(shrink(on_grey(Image.open(os.path.join(PROJECT, sources[i]))), 640))
        what = (f"the {steps[i]} step's picture" if i in steps else "made by you" if i in made else "from the user")
        labels.append(f"#{i + 1} ({what})")
    apose = (raw.get("step_pictures") or {}).get("00_apose")
    history = []
    for m in st["messages"][-14:]:
        pics = f" [sent pictures {', '.join(str(i + 1) for i in m['sources'])}]" if m.get("sources") else ""
        history.append(f"{'User' if m['who'] == 'you' else 'Slicer'}: {m.get('text', '')[:700]}{pics}")
    question = CHAT.format(base=slicer_plan.BASE, alone=slicer_plan.ALONE,
        apose=f"#{apose + 1}" if apose is not None else "not made yet (then use the newest picture)",
        pictures=", ".join(labels) or "none yet", character=raw.get("character") or name,
        parts=", ".join(f"{i + 1}. {p['name']}" for i, p in enumerate(parts)) or "not planned yet",
        step=f"{st['step'] + 1}. {step['name']}: {step.get('what', '')}" if step else "none",
        prompt=step.get("prompt", "") if step else "none",
        waiting=", ".join(f"{r['part']} ({r['do']})" for r in st["waiting"]) or "nothing",
        history="\n".join(history))
    m, processor = model()
    data, text = ask(m, processor, images, question, max_tokens=700)
    said = str(data.get("reply") or "").strip() or text.strip()[:1500]
    new_prompt = str(data.get("new_prompt") or "").strip()
    if new_prompt and "Locked:" not in new_prompt and step:  # the locks always stay
        new_prompt += " " + slicer_plan.LOCK
    wish = data.get("picture") if isinstance(data.get("picture"), dict) else {}
    instruction = str(wish.get("instruction") or "").strip()
    answer(name, upto, said, new_prompt or None)
    print(time.strftime("%H:%M:%S"), "answered", name, "(prompt changed)" if new_prompt else "",
          "(picture)" if instruction else "", flush=True)
    if instruction:
        try:
            start = int(wish.get("from") or len(sources)) - 1
        except (TypeError, ValueError):
            start = len(sources) - 1
        start = min(max(start, 0), len(sources) - 1) if sources else -1
        if apose is not None and (wish.get("only") or "mannequin with exactly" in instruction or "Remove all the clothes" in instruction):
            start = apose  # a base body or a piece alone: from the A-pose, so all layers fit together (ryan, Nov 24)
        if start < 0:
            slicer_says(name, "Send me a picture first (📎), then I can change it.")
        else:
            return start, instruction, str(wish.get("only") or "").strip()  # made by the caller, once Qwen3-VL is let go
    return None


DOLL = """Look closely at this chest. Are nipples visible (small raised points, dots or darker circles at the
tips)? Answer with JSON only: {"nipples": true or false}."""


def doll_ok(picture):
    """Is the base body a smooth mannequin (no nipples)? The edit model doesn't always obey ("no nipples" gave nipples
    on ryan's skin, Nov 24). Qwen3-VL looks at a close-up of the chest (asked about the whole body it said "nipples" to
    every body with a chest shape, smooth or not), and a failed picture is never shown."""
    import numpy as np
    from organise_pictures import ask
    from describe_colours import on_grey
    from scipy import ndimage
    rgba = picture.convert("RGBA")
    alpha = np.asarray(rgba.getchannel("A")) > 128
    labels, n = ndimage.label(alpha)
    if n == 0:
        return False
    biggest = labels == (1 + int(np.argmax(ndimage.sum(alpha, labels, range(1, n + 1)))))  # one figure (a sheet has two)
    ys, xs = np.nonzero(biggest)
    top, h = ys.min(), ys.max() - ys.min()
    cx = (xs.min() + xs.max()) / 2
    chest = on_grey(rgba).crop((int(cx - 0.14 * h), int(top + 0.20 * h), int(cx + 0.14 * h), int(top + 0.42 * h)))
    chest = chest.resize((512, round(512 * chest.height / max(chest.width, 1))))
    m, processor = model()
    data, text = ask(m, processor, [chest], DOLL, max_tokens=30)
    ok = data.get("nipples") is False
    print(time.strftime("%H:%M:%S"), "mannequin check:", "ok" if ok else f"FAILED {data or text[:80]}", flush=True)
    return ok


def base_body(source, instruction, seed=1, only=None, keep_frame=True):
    """A doll base body that passed the check: up to two tries, then plain underwear. Returns (raw, clear, grey,
    note): note says what happened when it wasn't the first try."""
    import slicer_pictures
    for k in range(2):
        let_go(quiet=False)
        raw_picture = slicer_pictures.edit(source, instruction, seed + k)
        clear, grey = slicer_pictures.cut_out(raw_picture, only=only, keep_frame=keep_frame)
        if doll_ok(clear):
            return raw_picture, clear, grey, ("" if k == 0 else "(the first try showed body details, so I made it again)")
    let_go(quiet=False)
    raw_picture = slicer_pictures.edit(source, slicer_plan.BASE_UNDERWEAR, seed)
    clear, grey = slicer_pictures.cut_out(raw_picture, only=only, keep_frame=keep_frame)
    return raw_picture, clear, grey, ("(the smooth doll body came out with body details twice, so this one is in plain "
                                      "grey underwear instead)")


def running(name, on):
    path = os.path.join(ROOT, name, "edit_running.json")
    if on:
        open(path, "w").write("{}")
    elif os.path.isfile(path):
        os.remove(path)


def make_picture(name, start, instruction, only=""):
    """A picture edited from the user's picture number start+1, added as the next numbered picture."""
    import slicer_pictures
    from PIL import Image
    raw = json.load(open(os.path.join(ROOT, name, "slicer.json")))
    slicer_says(name, f"Making it from picture {start + 1} (about a minute)…", part_id="_edit", kind="working")
    running(name, True)
    t = time.time()
    try:
        let_go(quiet=False)  # Qwen3-VL leaves the GPU while the picture is made: room on a 24 GB card too
        source = Image.open(os.path.join(PROJECT, raw["sources"][start]))
        square = source.width == source.height  # made pictures are square: the result keeps their frame
        note = ""
        if "mannequin with exactly" in instruction or "doll body" in instruction:
            raw_picture, clear, grey, note = base_body(source, instruction, keep_frame=square)
        else:
            raw_picture = slicer_pictures.edit(source, instruction)
            clear, grey = slicer_pictures.cut_out(raw_picture, only=only or None, keep_frame=square)
        index = add_made(name, clear)
        grey.save(os.path.join(ROOT, name, "sources", f"{index + 1:02d}_made_grey.png"))
        raw_picture.save(os.path.join(ROOT, name, "sources", f"{index + 1:02d}_made_raw.png"))  # the edit model's own
        with open(os.path.join(ROOT, name, "edits.jsonl"), "a") as log:  # what was asked, for checking later
            log.write(json.dumps({"picture": index + 1, "from": start + 1, "instruction": instruction, "only": only,
                                  "seconds": round(time.time() - t)}, ensure_ascii=False) + "\n")
        slicer_says(name, f"Here it is: **#{index + 1}** ({slicer_pictures.SIZE} x {slicer_pictures.SIZE}, background "
                          f"cut out, {round(time.time() - t)} s){(' ' + note) if note else ''}. Tell me what to change, "
                          "or ask for the next one.", sources=[index])
        print(time.strftime("%H:%M:%S"), "picture made", name, index + 1, round(time.time() - t), "s", flush=True)
    finally:
        running(name, False)


def match_skin(out, glb):
    """The base body's 3D skin brought to its pictures' skin: one gain per colour channel on the texture, from the
    lit skin's median (de-lit and without MV-Adapter TRELLIS.2 made it darker and redder: 183,112,97 for 205,152,127)."""
    import numpy as np
    import trimesh
    from PIL import Image
    pics = [np.asarray(Image.open(os.path.join(out, f"{n}.png")).convert("RGBA")).astype(np.float32)
            for n in ("colour_clear", "colour_back_clear") if os.path.isfile(os.path.join(out, f"{n}.png"))]
    px = np.concatenate([p[..., :3][p[..., 3] > 200] for p in pics])
    lum = px.mean(1)
    target = np.median(px[(lum > np.percentile(lum, 40)) & (lum < np.percentile(lum, 95))], 0)
    scene = trimesh.load(glb, force="scene")
    for g in scene.geometry.values():
        mat = g.visual.material
        tex = getattr(mat, "baseColorTexture", None)
        if tex is None:
            continue
        a = np.asarray(tex.convert("RGB")).astype(np.float32)
        used = a.sum(-1) > 30
        l2 = a[used].mean(1)
        med = np.median(a[used][(l2 > np.percentile(l2, 40)) & (l2 < np.percentile(l2, 95))], 0)
        mat.baseColorTexture = Image.fromarray(np.clip(a * np.clip(target / med, 0.5, 2.5), 0, 255).astype(np.uint8))
    scene.export(glb, file_type="glb")
    print(time.strftime("%H:%M:%S"), "skin matched to", target.round(), flush=True)


def make_3d(name, part_id):
    """Good, make it 3D: the part's transparent coloured and grey pictures through make_model.py, the full pipeline
    (de-light, TRELLIS.2 at the machine's resolution, MV-Adapter colours merged: about 20 to 30 minutes); the result
    is the part's model.glb, which the page shows on the right by itself."""
    import shutil
    import subprocess
    out = os.path.join(ROOT, name, "parts", part_id)
    colour, grey = os.path.join(out, "colour_clear.png"), os.path.join(out, "grey_clear.png")
    colours, greys = [colour], [grey]
    if os.path.isfile(os.path.join(out, "colour_back_clear.png")) and os.path.isfile(os.path.join(out, "grey_back_clear.png")):
        colours.append(os.path.join(out, "colour_back_clear.png")); greys.append(os.path.join(out, "grey_back_clear.png"))
    if not (os.path.isfile(colour) and os.path.isfile(grey)):
        slicer_says(name, "This part has no pictures yet: press Make it first.", part_id=part_id)
        return
    let_go()  # everything off the GPU: TRELLIS.2 needs it alone
    print(time.strftime("%H:%M:%S"), "3D", name, part_id, flush=True)
    t = time.time()
    work = os.path.join(out, "3d")
    base = part_id == "01_base_body"
    if os.path.isdir(work):  # a new try: the old stages would be skipped as "already there"
        shutil.rmtree(work)
    run = subprocess.run([sys.executable, os.path.join(HERE, "make_model.py"), "--pictures", *colours, "--grey", *greys,
                          "--out", work, *(["--no-mvadapter"] if base else [])],  # full pipeline (the user, Nov 24);
                         # the base body's even skin without MV-Adapter: its merge made it blotchy
                         capture_output=True, text=True, cwd=PROJECT)
    open(os.path.join(out, "3d_log.txt"), "w").write(run.stdout + run.stderr)
    done = [line for line in run.stdout.splitlines() if line.startswith("done in")]
    if run.returncode != 0 or not done:
        last = (run.stderr.strip().splitlines() or run.stdout.strip().splitlines() or ["?"])[-1]
        slicer_says(name, f"The 3D model didn't work: {last[:250]}\nThe log is parts/{part_id}/3d_log.txt.", part_id=part_id)
        return
    result = done[-1].split(": ", 1)[1].strip()
    shutil.copy(result, os.path.join(out, "model.glb.tmp"))
    if base:
        try:
            match_skin(out, os.path.join(out, "model.glb.tmp"))
        except Exception as e:
            print(time.strftime("%H:%M:%S"), "skin not matched:", repr(e), flush=True)
    os.replace(os.path.join(out, "model.glb.tmp"), os.path.join(out, "model.glb"))  # the chat announces it
    print(time.strftime("%H:%M:%S"), "3D made", name, part_id, round((time.time() - t) / 60, 1), "min", flush=True)
    try:
        place(name, part_id)
    except Exception as e:  # without a place the part is still shown, side by side
        print(time.strftime("%H:%M:%S"), "no place for", part_id, repr(e), flush=True)


def place(name, part_id):
    """Where the part sits on the body, for the page's Assembled view (parts/<id>/place.json): every layer keeps the
    A-pose's frame, so the piece's box in its own picture, against the whole figure's box in the A-pose picture, is
    its place: in figure heights from the figure's centre (x right, y down), and its height and width."""
    import numpy as np
    from PIL import Image
    folder = os.path.join(ROOT, name)
    out = os.path.join(folder, "parts", part_id, "place.json")
    def box(path):
        alpha = np.asarray(Image.open(path).getchannel("A")) > 128
        ys, xs = np.nonzero(alpha)
        return (xs.min(), xs.max(), ys.min(), ys.max()) if len(xs) else None
    apose = os.path.join(folder, "parts", "00_apose", "colour_clear.png")
    mine = os.path.join(folder, "parts", part_id, "colour_frame.png")  # in place (colour_clear.png is filled)
    if not os.path.isfile(mine):
        mine = os.path.join(folder, "parts", part_id, "colour_clear.png")
    if os.path.isfile(os.path.join(folder, "parts", part_id, "own_frame.json")):  # not from the A-pose: by hand
        if os.path.isfile(out):
            os.remove(out)
        return
    if part_id == "00_apose" or not (os.path.isfile(apose) and os.path.isfile(mine)):  # the base body too is placed
        json.dump({"cx": 0, "cy": 0, "h": 1, "w": None, "whole": True}, open(out, "w"))
        return
    f, p = box(apose), box(mine)
    if f is None or p is None:
        return
    fh, fcx, fcy = f[3] - f[2], (f[0] + f[1]) / 2, (f[2] + f[3]) / 2
    json.dump({"cx": float(((p[0] + p[1]) / 2 - fcx) / fh), "cy": float(((p[2] + p[3]) / 2 - fcy) / fh),
               "h": float((p[3] - p[2]) / fh), "w": float((p[1] - p[0]) / fh)}, open(out, "w"))


BOXES = ('Find {what} in this picture. Answer with JSON only: {{"boxes": [[x0, y0, x1, y1], ...]}}: one box around '
         'each separate piece of it (a pair of boots: two boxes), in 0-1000 coordinates of the picture.')


def boxes_for(picture, words):
    """Qwen3-VL's boxes (pixels of `picture`) around each of `words` (several words: the union)."""
    from organise_pictures import ask
    def to_pixels(b, width, height, shown):  # organise_pictures.to_pixels drops boxes under 5 %: a glass, a ring
        try:
            x0, y0, x1, y1 = (float(v) for v in b)
        except (TypeError, ValueError):
            return None
        scale = 1000.0 if max(x0, y0, x1, y1) <= 1000 else float(max(shown))
        x0, x1 = sorted((x0 / scale * width, x1 / scale * width))
        y0, y1 = sorted((y0 / scale * height, y1 / scale * height))
        if x1 - x0 < 4 or y1 - y0 < 4:
            return None
        pad = 0.06 * max(x1 - x0, y1 - y0)
        return (max(0, int(x0 - pad)), max(0, int(y0 - pad)), min(width, int(x1 + pad)), min(height, int(y1 + pad)))
    m, processor = model()
    small = picture.convert("RGB").resize((1024, 1024))
    found = []
    for word in words:
        data, text = ask(m, processor, [small], BOXES.format(what=word), max_tokens=200)
        if not data.get("boxes"):
            print(time.strftime("%H:%M:%S"), "no boxes for", repr(word), "answer:", text[:200].replace("\n", " "), flush=True)
        for b in data.get("boxes") or []:
            px = to_pixels(b, picture.width, picture.height, (1024, 1024))
            if px is not None:
                found.append([int(v) for v in px])
    return found


def piece_boxes(name):
    """Every piece's boxes on the A-pose picture, found once for all (parts[*]["boxes"] in slicer.json). Qwen3-VL
    sometimes adds another piece's boxes to a piece (demon's boot boxes came with the corset, the cape and the head):
    a box found for several pieces stays with the one whose boxes it makes up the largest share of."""
    from PIL import Image
    folder = os.path.join(ROOT, name)
    raw = json.load(open(os.path.join(folder, "slicer.json")))
    apose = Image.open(os.path.join(folder, "parts", "00_apose", "colour_clear.png"))
    on_grey = Image.new("RGB", apose.size, (128, 128, 128))
    on_grey.paste(apose, (0, 0), apose)
    pieces = [p for p in raw["parts"] if p.get("kind") == "piece"]
    found = {p["id"]: boxes_for(on_grey, p.get("words") or [p["what"]]) for p in pieces}
    same = lambda a, b: all(abs(x - y) < 0.02 * apose.width for x, y in zip(a, b))
    for pid, boxes in found.items():
        keep = []
        for box in boxes:
            owners = [q for q, other in found.items() if any(same(box, o) for o in other)]
            share = {q: sum(any(same(o, x) for x in found[q]) for o in found[q] if same(o, box)) / len(found[q]) for q in owners}
            if max(share, key=share.get) == pid:
                keep.append(box)
        found[pid] = keep or boxes
    for p in raw["parts"]:
        if p["id"] in found:
            p["boxes"] = found[p["id"]]
    json.dump(raw, open(os.path.join(folder, "slicer.json.tmp"), "w"), indent=1, ensure_ascii=False)
    os.replace(os.path.join(folder, "slicer.json.tmp"), os.path.join(folder, "slicer.json"))
    print(time.strftime("%H:%M:%S"), "boxes", name, {k: len(v) for k, v in found.items()}, flush=True)
    return found


def kind_of(part):
    return part.get("kind") or {"00_apose": "apose", "01_base_body": "base"}.get(part["id"], "legacy")


def make_back(part, kind, out, colour_clear, grey_clear, request, seed, apose_path):
    """The layer's back view, coloured and grey, in the same frame as its filled front pictures."""
    import slicer_pictures as sp
    from PIL import Image, ImageOps
    let_go(quiet=False)
    if kind == "base":  # the mannequin turned round, then the same skin as the front
        grey_back_clear, grey_back = sp.back(grey_clear, "smooth matte grey mannequin with exactly this body "
                                             "shape, in the same A-pose", seed)
        g = ImageOps.grayscale(grey_back_clear.convert("RGB")).convert("RGBA"); g.putalpha(grey_back_clear.getchannel("A"))
        grey_back_clear = g
        tone = sp.skin_tone(Image.open(apose_path))
        colour_back_clear, _ = sp.tint(grey_back_clear, tone) if tone is not None else (grey_back_clear, None)
        let_go(quiet=False)
        real_raw = sp.edit(Image.composite(colour_back_clear.convert("RGB"), Image.new("RGB", colour_back_clear.size, sp.GREY),
                                           colour_back_clear.getchannel("A")), sp.SKIN, seed)
        real_clear, _ = sp.cut_out(real_raw, keep_frame=True)
        if doll_ok(real_clear):
            colour_back_clear = real_clear
    else:
        colour_back_clear, _ = sp.back(colour_clear, part.get("what") or part["name"], seed)
        let_go(quiet=False)
        grey_back_clear, _ = sp.redo(colour_back_clear, request["grey_prompt"], seed, match="shape", room=1.2)
        g = ImageOps.grayscale(grey_back_clear.convert("RGB")).convert("RGBA"); g.putalpha(grey_back_clear.getchannel("A"))
        grey_back_clear = g
    colour_back_clear.save(os.path.join(out, "colour_back_clear.png"))
    grey_back_clear.save(os.path.join(out, "grey_back_clear.png"))


def make_step(name, part_id):
    """The Make it / Make it again button, by the part's kind:
    apose  the whole character in an A-pose from the uploaded picture (the reference; no 3D)
    base   the A-pose's own body as a grey mannequin (checked), then in the skin's colour from the A-pose
    piece  cut out of the A-pose by its boxes and completed where it was hidden (slicer_pictures.piece)
    head   the A-pose made bald, the head and neck cut out of it
    then the same as grey clay, lined up on the A-pose's frame like the colour."""
    import slicer_pictures as sp
    from PIL import Image, ImageOps
    folder = os.path.join(ROOT, name)
    request_file = os.path.join(folder, "parts", part_id, "request.json")
    request = json.load(open(request_file))
    raw = json.load(open(os.path.join(folder, "slicer.json")))
    part = next(p for p in raw["parts"] if p["id"] == part_id)
    kind = kind_of(part)
    out = os.path.join(folder, "parts", part_id)
    try:
        if request["do"] == "make_back":
            if not os.path.isfile(os.path.join(out, "colour_clear.png")):
                slicer_says(name, "This part has no pictures yet: press Make it first.", part_id=part_id)
                return
            print(time.strftime("%H:%M:%S"), "back view", name, part_id, flush=True)
            make_back(part, kind, out, Image.open(os.path.join(out, "colour_clear.png")),
                      Image.open(os.path.join(out, "grey_clear.png")), request, 1,
                      os.path.join(folder, "parts", "00_apose", "colour_clear.png"))
            os.utime(os.path.join(out, "colour.png"))  # the chat shows the pictures again, with the back
            print(time.strftime("%H:%M:%S"), "back made", name, part_id, flush=True)
            return
        if request["do"] == "make_3d":
            if part.get("no3d"):
                slicer_says(name, "The A-pose is the reference the layers are cut from: it isn't made in 3D.", part_id=part_id)
                return
            return make_3d(name, part_id)
        print(time.strftime("%H:%M:%S"), "making", name, part_id, kind, flush=True)
        seed = 1 + sum(1 for f in os.listdir(out) if f.startswith("colour_raw"))
        apose_path = os.path.join(folder, "parts", "00_apose", "colour_clear.png")
        if kind != "apose" and not os.path.isfile(apose_path):
            slicer_says(name, "Make the A-pose first: every layer is cut from it.", part_id=part_id)
            return
        grey_clear = None
        if kind == "apose" or kind == "legacy":
            let_go(quiet=False)
            source = Image.open(os.path.join(PROJECT, raw["sources"][0]) if kind == "apose" else apose_path)
            raw_colour = sp.edit(source, request["prompt"], seed)
            raw_colour.save(os.path.join(out, "colour_raw.png"))
            colour_clear, colour = sp.cut_out(raw_colour, keep_frame=kind != "apose")
            let_go(quiet=False)
            raw_grey = sp.edit(colour, request["grey_prompt"], seed)
            raw_grey.save(os.path.join(out, "grey_raw.png"))
            grey_clear, grey = sp.cut_out(raw_grey, keep_frame=True)
        elif kind == "base":
            apose = Image.open(apose_path)
            raw_colour, mannequin, _, note = base_body(apose, request["prompt"], seed)
            raw_colour.save(os.path.join(out, "colour_raw.png"))
            if note:
                slicer_says(name, f"Base body: {note}", part_id=part_id)
            let_go(quiet=False)
            tone = sp.skin_tone(apose)
            colour_clear, colour = sp.tint(mannequin, tone) if tone is not None else (mannequin, None)
            if colour is None:
                colour = Image.new("RGB", mannequin.size, sp.GREY); colour.paste(mannequin, (0, 0), mannequin)
            else:  # realistic skin on the tinted mannequin, kept only if it stays smooth (checked like the mannequin)
                for k in range(2):
                    let_go(quiet=False)
                    real_raw = sp.edit(colour, sp.SKIN, seed + k)
                    real_clear, real = sp.cut_out(real_raw, keep_frame=True)
                    if doll_ok(real_clear):
                        colour_clear, colour = real_clear, real
                        real_raw.save(os.path.join(out, "skin_raw.png"))
                        break
                else:
                    slicer_says(name, "Base body: the realistic skin came out with body details twice, so it keeps "
                                      "the plain skin colour.", part_id=part_id)
            g = ImageOps.grayscale(mannequin.convert("RGB")).convert("RGBA")
            g.putalpha(mannequin.getchannel("A"))
            grey_clear = g
            grey = Image.new("RGB", g.size, sp.GREY); grey.paste(g, (0, 0), g)
        elif kind == "piece":
            boxes = part.get("boxes")
            if boxes is None:
                boxes = piece_boxes(name).get(part_id, [])
            let_go(quiet=False)
            made = sp.piece(Image.open(apose_path), boxes, part["what"], seed) if boxes else None
            own = False
            if made is None:
                # not on the A-pose (held items are left out of it, and the A-pose edit sometimes drops a piece:
                # demon's cape): cut from the uploaded picture instead; it doesn't line up by itself
                # the uploaded picture itself (square on grey): its background cut-out first lost demon's glass
                on_grey = sp.square_on_grey(Image.open(os.path.join(PROJECT, raw["sources"][0]))).resize(
                    (sp.SIZE, sp.SIZE), Image.LANCZOS)  # boxes in the same pixels the cut uses
                source = on_grey.convert("RGBA")
                boxes = boxes_for(on_grey, part.get("words") or [part["what"]])
                print(time.strftime("%H:%M:%S"), "boxes on the uploaded picture", part_id, boxes, flush=True)
                let_go(quiet=False)
                made = sp.piece(source, boxes, part["what"], seed) if boxes else None
                own = made is not None
            if made is None:
                slicer_says(name, f"I couldn't find **{part['name']}** on the A-pose or on your picture. Tell me where "
                                  "it is (for example: in the right hand), or skip it.", part_id=part_id)
                return
            if own:
                slicer_says(name, f"**{part['name']}** isn't on the A-pose picture, so I cut it from your picture: it "
                                  "won't sit in place by itself. In the 3D view use Move to put it where it belongs.",
                            part_id=part_id)
                json.dump({"own_frame": True}, open(os.path.join(out, "own_frame.json"), "w"))
            elif os.path.isfile(os.path.join(out, "own_frame.json")):
                os.remove(os.path.join(out, "own_frame.json"))
            colour_clear, colour, the_cut = made
            the_cut.save(os.path.join(out, "cut.png"))
        elif kind == "head":
            let_go(quiet=False)
            bald_raw = sp.edit(Image.open(apose_path), request["prompt"], seed)
            bald_raw.save(os.path.join(out, "colour_raw.png"))
            bald, _ = sp.cut_out(bald_raw, keep_frame=True)
            boxes = boxes_for(bald, ["the head with the face and the neck"])
            let_go(quiet=False)
            head = sp.cut(bald, boxes[:1]) if boxes else None
            if head is None:
                slicer_says(name, "I couldn't find the head on the bald picture: try Make it again.", part_id=part_id)
                return
            head.save(os.path.join(out, "cut.png"))
            # SAM stops at the chin, leaving bits of the neck: completed like a piece, in place
            colour_clear, colour = sp.redo(head, sp.COMPLETE.format(piece="the bald head with the face and the whole neck"),
                                           seed, match="colour", room=1.5)
        if grey_clear is None:  # the same layer as grey clay, lined up by its outline
            let_go(quiet=False)
            grey_clear, grey = sp.redo(colour_clear, request["grey_prompt"], seed, match="shape", room=1.4)
        if kind == "piece" and os.path.isfile(os.path.join(out, "cut.png")) and not os.path.isfile(os.path.join(out, "own_frame.json")):
            # holes the cut piece had (a cut-out where skin showed, a wig's face opening) stay empty: told so in
            # words, the edit model still filled demon's heart cut-out with a red heart twice (Nov 24)
            import numpy as np
            from scipy import ndimage
            had = np.asarray(Image.open(os.path.join(out, "cut.png")).getchannel("A")) > 128
            holes = ndimage.binary_erosion(ndimage.binary_fill_holes(had) & ~had, iterations=3)
            if holes.any():
                for layer in (colour_clear, grey_clear):
                    if layer is not None:
                        a = np.asarray(layer.getchannel("A")).copy(); a[holes] = 0; layer.putalpha(Image.fromarray(a))
                colour = Image.new("RGB", colour_clear.size, sp.GREY); colour.paste(colour_clear, (0, 0), colour_clear)
        # the clay is grey for sure: the clay edit kept demon's red heart (Nov 24); only the light and shade stay
        g = ImageOps.grayscale(grey_clear.convert("RGB")).convert("RGBA")
        g.putalpha(grey_clear.getchannel("A"))
        grey_clear = g
        grey = Image.new("RGB", g.size, sp.GREY); grey.paste(g, (0, 0), g)
        if kind != "apose":  # in place for the placement, filled for the 3D and the eye
            colour_clear.save(os.path.join(out, "colour_frame.png"))
            grey_clear.save(os.path.join(out, "grey_frame.png"))
            colour_clear, colour, grey_clear, grey, box = sp.fill(colour_clear, grey_clear)
            json.dump({"box": box, "frame": sp.SIZE, "note": "where this picture sits on the A-pose: its square in "
                       "the A-pose picture's pixels"}, open(os.path.join(out, "frame.json"), "w"))
        if kind != "apose":  # the back view too: TRELLIS.2 then sees both sides (the base body's back was guessed)
            make_back(part, kind, out, colour_clear, grey_clear, request, seed, apose_path)
        colour_clear.save(os.path.join(out, "colour_clear.png"))
        grey_clear.save(os.path.join(out, "grey_clear.png"))
        grey.save(os.path.join(out, "grey.png"))
        colour.save(os.path.join(out, "colour.png"))  # last: the chat announces the pictures when colour.png changes
        print(time.strftime("%H:%M:%S"), "made", name, part_id, flush=True)
    finally:
        if os.path.isfile(request_file):
            os.remove(request_file)


def main():
    global _last
    print("The Slicer's worker: watching", ROOT, flush=True)
    while True:
        busy = False
        for name in sorted(os.listdir(ROOT)) if os.path.isdir(ROOT) else []:
            folder = os.path.join(ROOT, name)
            if not os.path.isfile(os.path.join(folder, "slicer.json")):
                continue
            try:
                if os.path.isfile(os.path.join(folder, "plan_request.json")):
                    plan(name); busy = True
                n = unanswered(name)
                if n is not None:
                    wish = reply(name, n); busy = True
                    if wish:
                        make_picture(name, *wish)
                parts_dir = os.path.join(folder, "parts")
                for part_id in sorted(os.listdir(parts_dir)) if os.path.isdir(parts_dir) else []:
                    if os.path.isfile(os.path.join(parts_dir, part_id, "request.json")):
                        make_step(name, part_id); busy = True
            except Exception as e:  # one job going wrong never stops the worker
                print(time.strftime("%H:%M:%S"), "ERROR", name, repr(e), flush=True)
                traceback.print_exc()
                try:
                    slicer_says(name, f"That didn't work on my side: {type(e).__name__}: {str(e)[:200]}")
                except Exception:
                    pass
                try:
                    n = unanswered(name)
                    if n is not None:
                        answer(name, n, f"Something went wrong on my side ({type(e).__name__}). Try again in a moment.")
                except Exception:
                    pass
        if busy:
            _last = time.time()
        elif (_model is not None or "slicer_pictures" in sys.modules) and _last and time.time() - _last > IDLE:
            let_go()
            _last = 0.0
        time.sleep(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
