"""The Slicer's web page: a character picture sliced into parts that are each made alone, seen as it grows.

Two halves (swappable): left the Slicer (the source picture, the locked settings, the slicer prompt, and the parts
in order, the centre always first, each with its prompt and its two reference pictures, coloured and grey), right the
3D environment (every part's model shown live as soon as it is there, each one on or off).

A project is a folder outputs/slicer/<name>/ with its plan, slicer.json, and per part parts/<id>/:
colour.png (the coloured reference), grey.png (the same as grey clay), model.glb (the part in 3D). The page asks
every two seconds and loads what is new, so a part made by a script (or by hand) appears by itself.

    python scripts/pipeline/slicer_page.py            # then open http://127.0.0.1:8780
    (on the uni server, from your PC: ssh -L 8780:127.0.0.1:8780 <server>, then the same address)

Only on this computer (127.0.0.1): it can change the plans.
"""
import argparse
import json
import mimetypes
import os
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


def state(name):
    """The plan with what is there of every part (each file's address carries its time: a new one is loaded)."""
    plan = json.load(open(plan_path(name)))
    folder = os.path.join(ROOT, name)
    def files(rel):
        found = {}
        for f in FILES:
            path = os.path.join(folder, rel, f)
            if os.path.isfile(path):
                found[f.split(".")[0]] = f"/files/{name}/{rel}/{f}?t={int(os.path.getmtime(path))}"
        return found
    for part in plan.get("parts", []):
        part["files"] = files(os.path.join("parts", part["id"]))
        part["status"] = ("made" if "model" in part["files"] else "pictures" if part["files"] else "planned")
    source = plan.get("source")
    if source and os.path.isfile(os.path.join(PROJECT, source)):
        plan["source_url"] = f"/source/{name}?t={int(os.path.getmtime(os.path.join(PROJECT, source)))}"
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
        """/api/part {project, id, field, value}: a part's prompt (or name, what) changed on the page."""
        try:
            data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            if urllib.parse.urlparse(self.path).path != "/api/part":
                return self.answer({"error": "no such page"}, 404)
            if data.get("field") not in EDITABLE:
                return self.answer({"error": f"{data.get('field')!r} can't be changed here"}, 400)
            path = plan_path(data.get("project", ""))
            plan = json.load(open(path))
            part = next((p for p in plan["parts"] if p["id"] == data.get("id")), None)
            if part is None:
                return self.answer({"error": f"no part {data.get('id')!r}"}, 404)
            part[data["field"]] = str(data.get("value", ""))
            json.dump(plan, open(path + ".tmp", "w"), indent=1, ensure_ascii=False)
            os.replace(path + ".tmp", path)  # whole or not at all
            return self.answer({"saved": True})
        except (ValueError, json.JSONDecodeError) as e:
            return self.answer({"error": str(e)}, 400)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--port", type=int, default=8780)
    a = p.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", a.port), Page)
    print(f"The Slicer: http://127.0.0.1:{a.port}   ({len(projects())} projects in {ROOT}; Ctrl+C stops it)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
