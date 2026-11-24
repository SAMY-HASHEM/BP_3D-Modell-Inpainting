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
import json
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


LOCK = threading.Lock()  # one writer at a time for the plans and the chats
ACTIONS = {"make_pictures": "Make it", "remake_pictures": "Make it again", "make_3d": "Good, make it 3D",
           "skip": "Skip this part"}


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
    names = ", ".join(p["name"] for p in plan["parts"])
    L = plan.get("locked", {})
    first = {"who": "slicer", "image": "source",
             "text": f"Here is {name}. I'll slice this character into {len(plan['parts'])} parts, and we make them "
                     f"together, one at a time: {names}.\n\nLocked for every picture: {L.get('pose', '')}, "
                     f"{L.get('size', '')}, {L.get('background', '')} background. Each part gets a coloured "
                     "picture and the same as grey clay, then its 3D model, shown on the right."}
    return {"messages": [first, step_message(plan, 0)], "step": 0, "announced": {}}


def load_chat(name, plan):
    path = chat_path(name)
    chat = json.load(open(path)) if os.path.isfile(path) else new_chat(name, plan)
    for i, m in enumerate(chat["messages"]):
        m.setdefault("n", i)
    return chat


def follow(name, plan, chat):
    """The Slicer's own messages: a part's pictures or model just appeared, then the next step."""
    changed = False
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
        chat["messages"].append({"who": "slicer", "text": "All parts are made. Switch the right half to Assembled to see them together."})
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
            "source_url": plan.get("source_url"), "waiting": waiting(name)}


def waiting(name):
    """What has been asked for and not made yet (a request file in the part's folder)."""
    folder = os.path.join(ROOT, name, "parts")
    found = []
    if os.path.isdir(folder):
        for pid in sorted(os.listdir(folder)):
            req = os.path.join(folder, pid, "request.json")
            if os.path.isfile(req):
                found.append(json.load(open(req)))
    return found


def say(name, text, action=None, part_id=None):
    """You wrote in the chat (or pressed a button): saved, and a button asks for the work (request.json)."""
    with LOCK:
        plan = state(name)
        chat = load_chat(name, plan)
        if action and action not in ACTIONS:
            raise ValueError(f"no action {action!r}")
        text = (text or "").strip() or ACTIONS.get(action, "")
        if not text:
            raise ValueError("an empty message")
        chat["messages"].append({"who": "you", "text": text[:4000], "time": time.strftime("%H:%M"),
                                 **({"action": action, "part": part_id} if action else {})})
        if action:
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
                             "make_3d": "Making the 3D model…"}[action]})
        save(chat_path(name), chat)


def slicer_says(name, text, part_id=None, images=()):
    """For the one doing the work (a script, or Claude in the terminal): a message from the Slicer in the chat."""
    with LOCK:
        plan = state(name)
        chat = load_chat(name, plan)
        chat["messages"].append({"who": "slicer", "text": text, **({"part": part_id} if part_id else {}),
                                 **({"images": list(images)} if images else {})})
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
    source = plan.get("source")
    if source and os.path.isfile(os.path.join(PROJECT, source)):
        plan["source_url"] = f"source/{name}?t={int(os.path.getmtime(os.path.join(PROJECT, source)))}"
    return plan


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
                return self.file(os.path.join(PROJECT, plan["source"]))
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
            if urllib.parse.urlparse(self.path).path == "/api/say":
                say(data.get("project", ""), data.get("text", ""), data.get("action"), data.get("part"))
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
        except (ValueError, json.JSONDecodeError) as e:
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
