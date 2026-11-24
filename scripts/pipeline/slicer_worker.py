"""The Slicer's worker: does what the page asks for, one job at a time on one GPU.

Every second it looks into outputs/slicer/*/:
- a plan_request.json: Qwen3-VL looks at the pictures and the parts are planned (slicer_plan.py);
- a message you typed that has no answer yet: Qwen3-VL answers it in the chat. It sees your pictures, the plan, the
  step you are on and the chat so far, and it can change the current step's prompt when you ask for that.
The pictures and the 3D models (parts/<id>/request.json) come next; until then they wait, and the chat says so.

Qwen3-VL stays loaded while you chat (about 17 GB on the GPU) and is let go after 10 quiet minutes.

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
from slicer_page import answer, chat_state, slicer_says, unanswered  # noqa: E402
import slicer_plan  # noqa: E402  (sets the model folder, offline)

IDLE = 600  # seconds without work before Qwen3-VL leaves the GPU
_model = None
_last = 0.0

CHAT = """You are the Slicer, the assistant inside a local app (no cloud) that turns a character picture into a 3D
model one part at a time: first the whole character in an A-pose, then the centre (the torso), then head, hair, arms,
legs, loose clothes and props. For every part the app makes a coloured picture and the same as grey clay (2048 x
2048, flat #808080 grey background), then a 3D model from them, shown on the right of the page.

You can see the user's pictures (in order: {pictures}). The character: {character}.
The parts: {parts}.
The step now: {step}
Its prompt for the image model now: {prompt}
Waiting for the image model: {waiting}
What works today: planning the parts, this chat, changing prompts. NOT connected yet: making the pictures and the 3D
models. A request for them only waits in a queue and nothing is made yet: never say you are making or will soon show
pictures; if the user asks for them, say plainly that this part is still being built.

The chat so far:
{history}

Answer the user's last message. Be short and concrete (at most 6 sentences), in the user's language, plain text.
If the user asks to change the picture of the step now (pose, what to keep or leave out, details), also write the
whole new prompt for it, in English, keeping the locked part ("Locked: ...") unchanged at its end.
Answer with JSON only: {{"reply": "...", "new_prompt": "" }}"""


def model():
    global _model
    if _model is None:
        from organise_pictures import load
        _model = load("Qwen/Qwen3-VL-8B-Instruct", False)
    return _model


def let_go():
    global _model
    if _model is not None:
        import torch
        _model = None
        gc.collect()
        torch.cuda.empty_cache()
        print(time.strftime("%H:%M:%S"), "Qwen3-VL let go (quiet)", flush=True)


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
    question = CHAT.format(
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
    answer(name, upto, said, new_prompt or None)
    print(time.strftime("%H:%M:%S"), "answered", name, "(prompt changed)" if new_prompt else "", flush=True)


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
                    reply(name, n); busy = True
            except Exception as e:  # one job going wrong never stops the worker
                print(time.strftime("%H:%M:%S"), "ERROR", name, repr(e), flush=True)
                traceback.print_exc()
                try:
                    n = unanswered(name)
                    if n is not None:
                        answer(name, n, f"Something went wrong on my side ({type(e).__name__}). Try again in a moment.")
                except Exception:
                    pass
        if busy:
            _last = time.time()
        elif _model is not None and time.time() - _last > IDLE:
            let_go()
        time.sleep(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
