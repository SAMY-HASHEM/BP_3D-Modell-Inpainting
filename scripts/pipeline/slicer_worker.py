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

You can see the user's pictures (in order: {pictures}). The character: {character}.
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
"{base}". Explain in "reply" that the base body
is made in plain underwear (the usual base for 3D clothes; no explicit pictures). A piece "alone" (a jacket, boots, ears, a
weapon): its instruction says "Show only <the piece> {alone}, seen from the front, keep its colours and details
exactly" (never the word "mannequin" alone: the model then draws a whole mannequin).
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
    for i, src in enumerate(sources[:4]):
        images.append(shrink(on_grey(Image.open(os.path.join(PROJECT, src))), 640))
        labels.append(f"picture {i + 1} from the user")
    if step:  # the step's own pictures, once made
        for kind in ("colour", "grey"):
            path = os.path.join(ROOT, name, "parts", step["id"], f"{kind}.png")
            if os.path.isfile(path) and len(images) < 6:
                images.append(shrink(on_grey(Image.open(path)), 640))
                labels.append(f"the {kind} picture made for this step")
    history = []
    for m in st["messages"][-14:]:
        pics = f" [sent pictures {', '.join(str(i + 1) for i in m['sources'])}]" if m.get("sources") else ""
        history.append(f"{'User' if m['who'] == 'you' else 'Slicer'}: {m.get('text', '')[:700]}{pics}")
    question = CHAT.format(base=slicer_plan.BASE, alone=slicer_plan.ALONE,
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
        if start < 0:
            slicer_says(name, "Send me a picture first (📎), then I can change it.")
        else:
            return start, instruction, str(wish.get("only") or "").strip()  # made by the caller, once Qwen3-VL is let go
    return None


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
        raw_picture = slicer_pictures.edit(source, instruction)
        clear, grey = slicer_pictures.cut_out(raw_picture, only=only or None)
        index = add_made(name, clear)
        grey.save(os.path.join(ROOT, name, "sources", f"{index + 1:02d}_made_grey.png"))
        raw_picture.save(os.path.join(ROOT, name, "sources", f"{index + 1:02d}_made_raw.png"))  # the edit model's own
        with open(os.path.join(ROOT, name, "edits.jsonl"), "a") as log:  # what was asked, for checking later
            log.write(json.dumps({"picture": index + 1, "from": start + 1, "instruction": instruction, "only": only,
                                  "seconds": round(time.time() - t)}, ensure_ascii=False) + "\n")
        slicer_says(name, f"Here it is: **picture {index + 1}** ({slicer_pictures.SIZE} x {slicer_pictures.SIZE}, "
                          f"background cut out, {round(time.time() - t)} s). Tell me what to change, or ask for the "
                          "next one.", sources=[index])
        print(time.strftime("%H:%M:%S"), "picture made", name, index + 1, round(time.time() - t), "s", flush=True)
    finally:
        running(name, False)


def make_3d(name, part_id):
    """Good, make it 3D: the part's transparent coloured and grey pictures through make_model.py (de-light, TRELLIS.2
    shape with its own colours, at 1024: about 6 to 10 minutes); the result is the part's model.glb, which the page
    shows on the right by itself."""
    import shutil
    import subprocess
    out = os.path.join(ROOT, name, "parts", part_id)
    colour, grey = os.path.join(out, "colour_clear.png"), os.path.join(out, "grey_clear.png")
    if not (os.path.isfile(colour) and os.path.isfile(grey)):
        slicer_says(name, "This part has no pictures yet: press Make it first.", part_id=part_id)
        return
    let_go()  # everything off the GPU: TRELLIS.2 needs it alone
    print(time.strftime("%H:%M:%S"), "3D", name, part_id, flush=True)
    t = time.time()
    work = os.path.join(out, "3d")
    run = subprocess.run([sys.executable, os.path.join(HERE, "make_model.py"), "--pictures", colour, "--grey", grey,
                          "--out", work, "--resolution", "1024", "--no-mvadapter"],
                         capture_output=True, text=True, cwd=PROJECT)
    open(os.path.join(out, "3d_log.txt"), "w").write(run.stdout + run.stderr)
    done = [line for line in run.stdout.splitlines() if line.startswith("done in")]
    if run.returncode != 0 or not done:
        last = (run.stderr.strip().splitlines() or run.stdout.strip().splitlines() or ["?"])[-1]
        slicer_says(name, f"The 3D model didn't work: {last[:250]}\nThe log is parts/{part_id}/3d_log.txt.", part_id=part_id)
        return
    result = done[-1].split(": ", 1)[1].strip()
    shutil.copy(result, os.path.join(out, "model.glb.tmp"))
    os.replace(os.path.join(out, "model.glb.tmp"), os.path.join(out, "model.glb"))  # the chat announces it
    print(time.strftime("%H:%M:%S"), "3D made", name, part_id, round((time.time() - t) / 60, 1), "min", flush=True)


def make_step(name, part_id):
    """The Make it / Make it again button: the step's coloured picture, then the same as grey clay."""
    import slicer_pictures
    from PIL import Image
    folder = os.path.join(ROOT, name)
    request_file = os.path.join(folder, "parts", part_id, "request.json")
    request = json.load(open(request_file))
    raw = json.load(open(os.path.join(folder, "slicer.json")))
    try:
        if request["do"] == "make_3d":
            return make_3d(name, part_id)
        print(time.strftime("%H:%M:%S"), "making", name, part_id, flush=True)
        let_go(quiet=False)
        base_path = os.path.join(folder, "parts", "00_apose", "colour.png")  # later parts start from the A-pose
        if part_id == "00_apose" or not os.path.isfile(base_path):
            base_path = os.path.join(PROJECT, raw["sources"][0])
        seed = 1 + sum(1 for f in os.listdir(os.path.join(folder, "parts", part_id)) if f.startswith("colour"))
        out = os.path.join(folder, "parts", part_id)
        raw_colour = slicer_pictures.edit(Image.open(base_path), request["prompt"], seed)
        raw_colour.save(os.path.join(out, "colour_raw.png"))
        only = None if part_id in ("00_apose", "01_base_body") else next(
            (p["name"] for p in raw["parts"] if p["id"] == part_id), None)  # a piece alone: nothing else stays
        colour_clear, colour = slicer_pictures.cut_out(raw_colour, only=only)
        raw_grey = slicer_pictures.edit(colour, request["grey_prompt"], seed)
        raw_grey.save(os.path.join(out, "grey_raw.png"))
        grey_clear, grey = slicer_pictures.cut_out(raw_grey, only=only)
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
