"""The Slicer's web page: a character picture sliced into parts that are each made alone, seen as it grows.

Two halves (swappable): left the Slicer (the source picture, the locked settings, the slicer prompt, and the parts
in order, the centre always first, each with its prompt and its two reference pictures, coloured and grey), right the
3D environment (every part's model shown live as soon as it is there, each one on or off).

A project is a folder outputs/slicer/<name>/ with its plan, slicer.json, and per part parts/<id>/:
colour.png (the coloured reference), grey.png (the same as grey clay), model.glb (the part in 3D). The page asks
every two seconds and loads what is new, so a part made by a script (or by hand) appears by itself.

    python scripts/pipeline/slicer_page.py            # then open http://127.0.0.1:8780
    on the uni server: https://nc.hcigroup.de/user/<your name>/proxy/8780/ (JupyterHub's login; jupyter-server-proxy)
    or from your PC: ssh -L 8780:127.0.0.1:8780 <server>, then http://127.0.0.1:8780

Every address in the page is relative, so it works behind such a path too.

Only on this computer (127.0.0.1): it can change the plans.
"""
import argparse
import base64
import json
import re
import mimetypes
import os
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PAGE = os.path.join(PROJECT, "user_interface_model", "slicer.html")
ROOT = os.path.join(PROJECT, "outputs", "slicer")
FILES = ("colour.png", "grey.png", "model.glb")  # what one part can have, in the order it is made
EDITABLE = ("prompt", "grey_prompt", "what", "name")  # what the page may change in a part


def plan_path(name):
    if not name or "/" in name or name.startswith("."):
        raise ValueError(f"no project {name!r}")
    path = os.path.join(ROOT, name, "slicer.json")
    if not os.path.isfile(path):
        raise ValueError(f"no project {name!r}")
    return path


def projects():
    if not os.path.isdir(ROOT):
        return []
    return sorted(n for n in os.listdir(ROOT) if os.path.isfile(os.path.join(ROOT, n, "slicer.json")))


class _Lock:
    """One writer at a time for the plans and the chats, across threads and across programs (the page server and
    the worker both write a chat: a file lock, outputs/slicer/.lock)."""
    def __init__(self):
        self.thread = threading.RLock()
        self.depth = 0

    def __enter__(self):
        import fcntl
        self.thread.acquire()
        if self.depth == 0:
            os.makedirs(ROOT, exist_ok=True)
            self.file = open(os.path.join(ROOT, ".lock"), "w")
            fcntl.flock(self.file, fcntl.LOCK_EX)
        self.depth += 1

    def __exit__(self, *exc):
        import fcntl
        self.depth -= 1
        if self.depth == 0:
            fcntl.flock(self.file, fcntl.LOCK_UN)
            self.file.close()
        self.thread.release()


LOCK = _Lock()
ACTIONS = {"make_pictures": "Make it", "remake_pictures": "Make it again", "make_3d": "Good, make it 3D",
           "skip": "Skip this part", "replan": "Plan the parts again", "restart": "Start again from step 1"}


def chat_path(name):
    return os.path.join(os.path.dirname(plan_path(name)), "chat.json")


def save(path, data):
    json.dump(data, open(path + ".tmp", "w"), indent=1, ensure_ascii=False)
    os.replace(path + ".tmp", path)  # whole or not at all


def step_message(plan, index):
    part = plan["parts"][index]
    return {"who": "slicer", "part": part["id"], "kind": "step",
            "text": f"Step {index + 1} of {len(plan['parts'])}: **{part['name']}**\n{part.get('what', '')}\n\n"
                    "This is the prompt I'll give the image model. Change it below if you want, then press Make it.",
            "buttons": ["make_pictures", "skip"]}


def new_chat(name, plan):
    if not plan.get("sources") and not plan.get("source"):
        return {"messages": [{"who": "slicer", "text": f"A new character: **{name}**.\nSend me one or more pictures of "
                              "it with 📎 below (or drop or paste them here). Front, back, coloured or grey: all help. "
                              "Write anything I should know with them."}],
                "step": 0, "announced": {}, "planned": False}
    if not plan["parts"]:
        n = len(plan.get("sources") or [1])
        return {"messages": [{"who": "slicer", "image": "sources",
                              "text": f"Got your {n} picture{'s' if n > 1 else ''}. I'm looking at "
                                      f"{'them' if n > 1 else 'it'} to plan the parts (about a minute)…",
                              "kind": "working", "part": "_plan"}],
                "step": 0, "announced": {}, "planned": False}
    return {**planned_messages(name, plan), "step": 0, "announced": {}, "planned": True}


def planned_messages(name, plan):
    names = ", ".join(p["name"] for p in plan["parts"])
    L = plan.get("locked", {})
    who = f" ({plan['character']})" if plan.get("character") else ""
    first = {"who": "slicer", "image": "source",
             "text": f"Here is {name}{who}. I'll slice this character into {len(plan['parts'])} parts, and we make them "
                     f"together, one at a time: {names}.\n\nLocked for every picture: {L.get('pose', '')}, "
                     f"{L.get('size', '')}, {L.get('background', '')} background. Each part gets a coloured "
                     "picture and the same as grey clay, then its 3D model, shown on the right."}
    return {"messages": [first, step_message(plan, 0)]}


def load_chat(name, plan):
    path = chat_path(name)
    chat = json.load(open(path)) if os.path.isfile(path) else new_chat(name, plan)
    chat.setdefault("planned", bool(plan["parts"]))
    for i, m in enumerate(chat["messages"]):
        m.setdefault("n", i)
    return chat


def follow(name, plan, chat):
    """The Slicer's own messages: a part's pictures or model just appeared, then the next step."""
    changed = False
    if not chat["planned"] and plan["parts"]:  # the plan has just been made
        chat["planned"] = True
        chat["messages"].extend(planned_messages(name, plan)["messages"])
        return True
    for index, part in enumerate(plan["parts"]):
        seen = chat["announced"].setdefault(part["id"], {})
        files = part["files"]
        stamp = {k: v.split("?t=")[-1] for k, v in files.items()}
        if "colour" in files and seen.get("colour") != stamp["colour"]:
            seen["colour"] = stamp["colour"]; changed = True
            chat["messages"].append({"who": "slicer", "part": part["id"],
                "text": f"The pictures for **{part['name']}** are ready: coloured and grey. Are they right?",
                "images": [files["colour"]] + ([files["grey"]] if "grey" in files else []),
                "buttons": ["make_3d", "remake_pictures"]})
        if "model" in files and seen.get("model") != stamp["model"]:
            seen["model"] = stamp["model"]; changed = True
            chat["messages"].append({"who": "slicer", "part": part["id"],
                "text": f"**{part['name']}** is in 3D now: it's on the right. Turn it with the mouse to check it."})
            if index == chat["step"]:
                changed |= next_step(plan, chat)
    return changed


def next_step(plan, chat):
    if chat["step"] + 1 < len(plan["parts"]):
        chat["step"] += 1
        chat["messages"].append(step_message(plan, chat["step"]))
    else:
        chat["step"] = len(plan["parts"])
        made = [p["name"] for p in plan["parts"] if p.get("status") == "made"]
        pictured = [p["name"] for p in plan["parts"] if p.get("status") == "pictures"]
        missing = [p["name"] for p in plan["parts"] if p.get("status") == "planned"]
        lines = ["That was the last step."]
        if made:
            lines.append(f"In 3D: {', '.join(made)}. Switch the right half to Assembled to see them together.")
        if pictured:
            lines.append(f"Pictures only (no 3D yet): {', '.join(pictured)}.")
        if missing:
            lines.append(f"Skipped, nothing made yet: {', '.join(missing)}.")
        chat["messages"].append({"who": "slicer", "text": "\n".join(lines), "buttons": ["restart"], "part": "_plan"})
    return True


def chat_state(name):
    with LOCK:
        plan = state(name)
        chat = load_chat(name, plan)
        changed = follow(name, plan, chat) or not os.path.isfile(chat_path(name))
        if changed:
            save(chat_path(name), chat)
    for i, m in enumerate(chat["messages"]):
        m["n"] = i
    return {"messages": chat["messages"], "step": chat["step"], "parts": {p["id"]: p for p in plan["parts"]},
            "source_url": plan.get("source_url"), "source_urls": plan.get("source_urls", []), "waiting": waiting(name),
            "thinking": unanswered(name) is not None}


def waiting(name):
    """What has been asked for and not made yet (a request file in the part's folder)."""
    folder = os.path.join(ROOT, name, "parts")
    found = []
    if os.path.isfile(os.path.join(ROOT, name, "plan_request.json")):
        found.append({"part": "_plan", "do": "plan"})
    if os.path.isfile(os.path.join(ROOT, name, "edit_running.json")):
        found.append({"part": "_edit", "do": "edit"})
    if os.path.isdir(folder):
        for pid in sorted(os.listdir(folder)):
            req = os.path.join(folder, pid, "request.json")
            if os.path.isfile(req):
                found.append(json.load(open(req)))
    return found


def say(name, text, action=None, part_id=None, files=None):
    """You wrote in the chat (with pictures or not), or pressed a button: saved; a button asks for the work."""
    with LOCK:
        plan_file = plan_path(name)
        plan = state(name)
        chat = load_chat(name, plan)
        if action and action not in ACTIONS:
            raise ValueError(f"no action {action!r}")
        text = (text or "").strip() or ACTIONS.get(action, "")
        if not text and not files:
            raise ValueError("an empty message")
        message = {"who": "you", "text": text[:4000], "time": time.strftime("%H:%M"),
                   **({"action": action, "part": part_id} if action else {})}
        folder = os.path.dirname(plan_file)
        if files:
            raw = json.load(open(plan_file))
            sources = raw.get("sources") or []
            new = add_pictures(name, folder, files, sources)
            raw["sources"] = sources + new
            raw.setdefault("source", raw["sources"][0])
            save(plan_file, raw)
            message["sources"] = list(range(len(sources), len(sources) + len(new)))
            again = sorted(set(add_pictures.repeated))
            if again:
                message["repeated"] = again
        chat["messages"].append(message)
        if files and not message["sources"]:  # nothing new
            chat["messages"].append({"who": "slicer", "text": "I have " + ("this picture" if len(again) == 1 else "these pictures")
                                     + " already (picture " + ", ".join(str(i + 1) for i in again) + "), so I didn't add "
                                     + ("it" if len(again) == 1 else "them") + " again."})
            chat["answered"] = len(chat["messages"]) - 2
            save(chat_path(name), chat)
            return
        if files and not plan["parts"]:
            if not os.path.isfile(os.path.join(folder, "plan_request.json")):
                ask_plan(folder, text)
                chat["answered"] = len(chat["messages"]) - 1
                n = len(message["sources"])
                chat["messages"].append({"who": "slicer", "kind": "working", "part": "_plan",
                                         "text": f"Got {'them' if n > 1 else 'it'}. I'm looking at the "
                                                 f"picture{'s' if n > 1 else ''} to plan the parts (about a minute)…"})
        elif files and not text:  # no words with them: no need to ask the model
            chat["messages"].append({"who": "slicer", "buttons": ["replan"], "part": "_plan",
                                     "text": "I added the picture(s) to the references. Should I plan the parts again "
                                             "with them? (The parts already made stay where they are.)"})
            chat["answered"] = len(chat["messages"]) - 2
        if action == "restart":
            chat["step"] = 0
            chat["messages"].append(step_message(plan, 0))
        elif action == "replan":
            ask_plan(folder, text)
            raw = json.load(open(plan_file))
            raw["parts_before"] = raw.get("parts", [])
            raw["parts"] = []
            save(plan_file, raw)
            chat.update(planned=False, step=0, announced={})
            chat["messages"].append({"who": "slicer", "kind": "working", "part": "_plan",
                                     "text": "Planning the parts again…"})
        elif action:
            part = next((p for p in plan["parts"] if p["id"] == part_id), None)
            if part is None:
                raise ValueError(f"no part {part_id!r}")
            if action == "skip":
                chat["messages"].append({"who": "slicer", "text": f"Skipped **{part['name']}**."})
                if plan["parts"].index(part) == chat["step"]:
                    next_step(plan, chat)
            else:
                req = {"part": part["id"], "do": action, "prompt": part.get("prompt", ""),
                       "grey_prompt": part.get("grey_prompt", ""), "asked": time.strftime("%Y-%m-%d %H:%M:%S")}
                os.makedirs(os.path.join(ROOT, name, "parts", part["id"]), exist_ok=True)
                save(os.path.join(ROOT, name, "parts", part["id"], "request.json"), req)
                chat["messages"].append({"who": "slicer", "part": part["id"], "kind": "working",
                    "text": {"make_pictures": "Making the coloured and the grey picture…",
                             "remake_pictures": "Making the pictures again…",
                             "make_3d": "Making the 3D model (TRELLIS.2, about 6 to 10 minutes)…"}[action]})
        save(chat_path(name), chat)


def unanswered(name):
    """Your newest message that got no answer yet (typed text, not a button), or None."""
    path = chat_path(name)
    if not os.path.isfile(path):
        return None
    chat = json.load(open(path))
    done = chat.get("answered", -1)
    for i in range(len(chat["messages"]) - 1, done, -1):
        m = chat["messages"][i]
        if m["who"] == "you" and not m.get("action") and m.get("text"):
            return i
    return None


def answer(name, upto, text, new_prompt=None):
    """The worker's answer to your messages up to `upto`; with new_prompt, the current step's prompt changes."""
    with LOCK:
        plan = state(name)
        chat = load_chat(name, plan)
        chat["answered"] = max(chat.get("answered", -1), upto)
        message = {"who": "slicer", "text": text}
        theirs = chat["messages"][upto] if upto < len(chat["messages"]) else {}
        if theirs.get("sources") and plan["parts"]:
            message.update(buttons=["replan"], part="_plan")
        if new_prompt and chat["step"] < len(plan["parts"]):
            part_id = plan["parts"][chat["step"]]["id"]
            raw = json.load(open(plan_path(name)))
            for p in raw["parts"]:
                if p["id"] == part_id:
                    p["prompt"] = new_prompt
            save(plan_path(name), raw)
            message["text"] += "\n\nI changed the prompt of this step (you can see it above, under the step)."
        chat["messages"].append(message)
        save(chat_path(name), chat)


def add_made(name, picture):
    """A picture the Slicer made (a PIL picture): saved in sources/ as the next numbered picture; returns its index."""
    with LOCK:
        plan_file = plan_path(name)
        raw = json.load(open(plan_file))
        sources = raw.get("sources") or ([raw["source"]] if raw.get("source") else [])
        folder = os.path.join(os.path.dirname(plan_file), "sources")
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"{len(sources) + 1:02d}_made.png")
        picture.save(path)
        raw["sources"] = sources + [os.path.relpath(path, PROJECT)]
        raw.setdefault("source", raw["sources"][0])
        raw.setdefault("made", []).append(len(sources))
        save(plan_file, raw)
        return len(sources)


def slicer_says(name, text, part_id=None, images=(), kind=None, sources=None):
    """For the one doing the work (a script, or Claude in the terminal): a message from the Slicer in the chat."""
    with LOCK:
        plan = state(name)
        chat = load_chat(name, plan)
        chat["messages"].append({"who": "slicer", "text": text, **({"part": part_id} if part_id else {}),
                                 **({"images": list(images)} if images else {}), **({"kind": kind} if kind else {}),
                                 **({"sources": list(sources)} if sources else {})})
        save(chat_path(name), chat)


def state(name):
    """The plan with what is there of every part (each file's address carries its time: a new one is loaded)."""
    plan = json.load(open(plan_path(name)))
    folder = os.path.join(ROOT, name)
    def files(rel):
        found = {}
        for f in FILES:
            path = os.path.join(folder, rel, f)
            if os.path.isfile(path):
                found[f.split(".")[0]] = f"files/{name}/{rel}/{f}?t={int(os.path.getmtime(path))}"
        return found
    for part in plan.get("parts", []):
        part["files"] = files(os.path.join("parts", part["id"]))
        part["status"] = ("made" if "model" in part["files"] else "pictures" if part["files"] else "planned")
    sources = plan.get("sources") or ([plan["source"]] if plan.get("source") else [])
    plan["source_urls"] = [f"source/{name}?i={i}&t={int(os.path.getmtime(os.path.join(PROJECT, src)))}"
                           for i, src in enumerate(sources) if os.path.isfile(os.path.join(PROJECT, src))]
    if plan["source_urls"]:
        plan["source_url"] = plan["source_urls"][0]
    return plan


NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,39}$")
PICTURE = (".png", ".jpg", ".jpeg", ".webp")


def new_project(name, files):
    """A new project from uploaded pictures: sources/, a plan without parts, and the request to plan it."""
    name = (name or "").strip().lower().replace(" ", "_")
    if not NAME.match(name):
        raise ValueError("a name with letters, digits, - and _ only (at most 40)")
    folder = os.path.join(ROOT, name)
    if os.path.exists(folder):
        raise ValueError(f"there is a project {name!r} already")
    if files and len(files) > 12:
        raise ValueError("at most 12 pictures")
    os.makedirs(os.path.join(folder, "sources"))
    try:
        return made(name, folder, files)
    except Exception:
        import shutil
        shutil.rmtree(folder, ignore_errors=True)  # nothing half made stays
        raise


def made(name, folder, files):
    sources = add_pictures(name, folder, files, []) if files else []
    save(os.path.join(folder, "slicer.json"), {"name": name, "sources": sources,
                                                **({"source": sources[0]} if sources else {}), "parts": []})
    if sources:
        ask_plan(folder)
    return name


def add_pictures(name, folder, files, sources):
    """Uploaded pictures into sources/ (numbered after the ones there); checked to be real pictures."""
    from PIL import Image
    if len(files) + len(sources) > 12:
        raise ValueError("at most 12 pictures per character")
    os.makedirs(os.path.join(folder, "sources"), exist_ok=True)
    new = []
    repeated = add_pictures.repeated = []  # pictures sent before (their numbers), not added again
    for f in files:
        ext = os.path.splitext(str(f.get("name", "")))[1].lower()
        if ext not in PICTURE:
            raise ValueError(f"{f.get('name')!r} is not a picture (png, jpg, webp)")
        data = base64.b64decode(str(f.get("data", "")).split(",", 1)[-1])
        import hashlib
        digest = hashlib.sha1(data).hexdigest()
        have = {hashlib.sha1(open(os.path.join(PROJECT, x), "rb").read()).hexdigest(): i
                for i, x in enumerate(sources + new) if os.path.isfile(os.path.join(PROJECT, x))}
        if digest in have:
            repeated.append(have[digest])
            continue
        if len(data) > 30e6:
            raise ValueError(f"{f.get('name')!r} is over 30 MB")
        path = os.path.join(folder, "sources", f"{len(sources) + len(new) + 1:02d}{ext}")
        open(path, "wb").write(data)
        try:
            Image.open(path).verify()
        except Exception:
            os.remove(path)
            raise ValueError(f"{f.get('name')!r} could not be read as a picture")
        new.append(os.path.relpath(path, PROJECT))
    return new


def ask_plan(folder, note=""):
    save(os.path.join(folder, "plan_request.json"), {"do": "plan", "note": note, "asked": time.strftime("%Y-%m-%d %H:%M:%S")})


class Page(BaseHTTPRequestHandler):
    def log_message(self, *args):  # quiet: the page asks every two seconds
        pass

    def send(self, body, kind, code=200, extra=()):
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        for k, v in extra:
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def answer(self, data, code=200):
        self.send(json.dumps(data).encode(), "application/json", code, [("Cache-Control", "no-store")])

    def file(self, path):
        if not os.path.isfile(path):
            return self.answer({"error": "not there"}, 404)
        kind = "model/gltf-binary" if path.endswith(".glb") else mimetypes.guess_type(path)[0] or "application/octet-stream"
        self.send(open(path, "rb").read(), kind, extra=[("Cache-Control", "max-age=31536000, immutable")])

    def do_GET(self):
        address = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(address.query)
        try:
            if address.path == "/":
                return self.send(open(PAGE, "rb").read(), "text/html; charset=utf-8", extra=[("Cache-Control", "no-store")])
            if address.path == "/api/projects":
                return self.answer(projects())
            if address.path == "/api/project":
                return self.answer(state(query.get("name", [""])[0]))
            if address.path == "/api/chat":
                return self.answer(chat_state(query.get("name", [""])[0]))
            if address.path.startswith("/source/"):
                plan = json.load(open(plan_path(address.path.split("/")[2])))
                sources = plan.get("sources") or [plan["source"]]
                return self.file(os.path.join(PROJECT, sources[int(query.get("i", ["0"])[0]) % len(sources)]))
            if address.path.startswith("/files/"):
                name, rel = address.path[len("/files/"):].split("/", 1)
                folder = os.path.realpath(os.path.join(ROOT, name))
                path = os.path.realpath(os.path.join(folder, urllib.parse.unquote(rel)))
                if not path.startswith(folder + os.sep) or os.path.basename(path) not in FILES:
                    return self.answer({"error": "not allowed"}, 403)
                return self.file(path)
            return self.answer({"error": "no such page"}, 404)
        except ValueError as e:
            return self.answer({"error": str(e)}, 404)

    def do_POST(self):
        """/api/part {project, id, field, value}: a part's prompt (or name, what) changed on the page.
        /api/say {project, text, action?, part?}: a message in the chat, or a button pressed."""
        try:
            data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            if urllib.parse.urlparse(self.path).path == "/api/new":
                return self.answer({"name": new_project(data.get("name"), data.get("files"))})
            if urllib.parse.urlparse(self.path).path == "/api/say":
                say(data.get("project", ""), data.get("text", ""), data.get("action"), data.get("part"), data.get("files"))
                return self.answer({"saved": True})
            if urllib.parse.urlparse(self.path).path != "/api/part":
                return self.answer({"error": "no such page"}, 404)
            if data.get("field") not in EDITABLE:
                return self.answer({"error": f"{data.get('field')!r} can't be changed here"}, 400)
            path = plan_path(data.get("project", ""))
            plan = json.load(open(path))
            part = next((p for p in plan["parts"] if p["id"] == data.get("id")), None)
            if part is None:
                return self.answer({"error": f"no part {data.get('id')!r}"}, 404)
            with LOCK:
                plan = json.load(open(path))
                part = next(p for p in plan["parts"] if p["id"] == data.get("id"))
                part[data["field"]] = str(data.get("value", ""))
                save(path, plan)
            return self.answer({"saved": True})
        except (ValueError, json.JSONDecodeError, OSError) as e:
            return self.answer({"error": str(e)}, 400)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--port", type=int, default=8780)
    p.add_argument("--say", nargs=2, metavar=("PROJECT", "TEXT"), help="write in the chat as the Slicer, and stop")
    p.add_argument("--part", help="with --say: the part it is about")
    p.add_argument("--read", metavar="PROJECT", help="print the chat (what you wrote, and what is asked for), and stop")
    a = p.parse_args()
    if a.say:
        return slicer_says(a.say[0], a.say[1].replace("\\n", "\n"), a.part)
    if a.read:
        st = chat_state(a.read)
        for m in st["messages"][-15:]:
            print(f"[{m['n']}] {m['who']}: {m['text'][:300]}")
        for r in st["waiting"]:
            print("ASKED:", json.dumps(r, ensure_ascii=False)[:300])
        return
    server = ThreadingHTTPServer(("127.0.0.1", a.port), Page)
    print(f"The Slicer: http://127.0.0.1:{a.port}   ({len(projects())} projects in {ROOT}; Ctrl+C stops it)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
