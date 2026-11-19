"""The web page of the node flows: the node chain, every node's result pictures, before / after with a slider,
the edit by hand (place the part, paint the 2D mask) and the Run buttons.

A small server on this PC only (127.0.0.1). It reads what the flows in flows/ have made so far (the same nodes and
folders as flow.py, nodes_core.py) and shows it. "Run" starts flow.py for that flow (as in a terminal, its output
goes to outputs/<name>/work/run.log); a flow.py started in a terminal is followed the same way.

    python scripts/pipeline/flow_page.py            # then open http://127.0.0.1:8777
    python scripts/pipeline/flow_page.py --open     # opens the browser by itself

All pages are in user_interface_model/: http://127.0.0.1:8777/app is the main page (app.html: the chat, the data and the
city that shows the run); / is the node page (old/nodes.html), /old/app the main page before it (old/app.html).

For other people (e.g. on the uni server, inside the uni network): --accounts asks for a login first (user_interface_model/login.html).
The accounts are handed out with accounts.py, nobody makes one on the page. Only with --accounts may --host be
another address than this PC's own:

    python scripts/pipeline/accounts.py add anna                         # prints anna's password once
    python scripts/pipeline/flow_page.py --accounts --host 0.0.0.0       # everybody who reaches this computer, after a login

With --accounts: GET /login is the login page, POST /api/login {"name", "password"} logs in (a cookie),
POST /api/logout logs out, GET /api/me says who is logged in; everything else answers 401 without a login (the two
pages go to /login), and "Open in Blender" is off (it would open a window on the computer the page runs on).
Not yet: every account sees and changes the same projects, and two runs at once are not put in a row.
The page gets everything from here:
    GET /api/flows              the flow files in flows/
    GET /api/flow?name=<file>   one flow: its nodes, links, states, pictures, and what can be compared
    GET /api/server?flow=<file> the uni server jobs of this flow's nodes and how far they are (cloud.py status; only when
                                the page asks); ?all = every job in the cloud, that takes over a minute
    GET /file/<path>            a picture or video from outputs/, pictures/ or flows/ (nothing else is given out)
    POST /api/run               {"flow", "upto"}: start flow.py (upto = only this node and what it needs)
    POST /api/stop              {"flow"}: stop that flow.py (a job on the uni server goes on there and is waited for next time)
    POST /api/edit              {"flow", "node", "place", "side", "replace", "layers", "after", "run"}: the edit by hand; it
                                is written into the flow file (the node's settings) and flows/hand/ (the painted mask)
    POST /api/settings          {"flow", "node", "settings"}: a node's settings, written into the flow file
    POST /api/blender           {"model"}: opens a model of outputs/ in a new Blender window
    POST /api/prompt            {"flow", "node", "prompt", "seed", "reference", "area"}: the local picture AI paints the
                                prompt into the painted area of the main side (picture_from_prompt.py), into flows/hand/

The main page makes its own flow files with these (POST /api/understand {"prompt", "items"} before them: the local AI
reads the chat's words and data and says which data is the model and which the new part):
    POST /api/upload?flow=<name>&role=object|part&name=<file>   the request's body is the picture or video itself; it is
                                saved as pictures/uploads/<flow>/<role>/<file>
    POST /api/app/create        {"flow", "prompt", "items": [{"path", "role", "prompt"}], "part_prompt", "settings": {"seed",
                                "steps", "resolution", "where"}}: writes flows/<flow>.json from what was uploaded: the object's
                                base model, and when there is a part the edit and the three inpainting steps. part_prompt:
                                the part in words, when there is no picture of it (the edit's "prompt")
    GET /file/<path>.glb        a model out of outputs/, for the 3D view on the main page
"""
import argparse
import base64
import json
import mimetypes
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import accounts
import flow as flows
import nodes_core as core

UI = os.path.join(core.PROJECT, "user_interface_model")  # every page of the web site
PAGE = os.path.join(UI, "old", "nodes.html")  # the node page (GET /)
APP = os.path.join(UI, "app.html")  # the main page (GET /app): the chat, the data, the city
OLD_APP = os.path.join(UI, "old", "app.html")  # the main page before it (GET /old/app): data rows, the developer view
LOGIN = os.path.join(UI, "login.html")  # GET /login, only with --accounts
COOKIE = "flow_login"
UPLOADS = os.path.join(core.PROJECT, "pictures", "uploads")  # what the main page sends: uploads/<flow>/<role>/<file>
NAME = r"[A-Za-z0-9_-]{1,40}"  # a flow the main page makes: its name is a file name and a folder name
FLOWS = os.path.join(core.PROJECT, "flows")
HAND = os.path.join(FLOWS, "hand")  # the painted mask corrections of the edits by hand: hand/<flow>_<node>/add_<view>.png
OPEN = ("outputs", "pictures", "flows")  # the only folders the page gets files from
ENV_PYTHON = os.path.join(core.PROJECT, "conda", "envs", "inpainting-3d-modelling", "bin", "python")  # activate.sh's Python
RUNS = {}  # {flow file: {"process", "upto", "log", "started"}}: the flow.py this page started
# what can be set on the page per node type, with the value a node has when the flow file doesn't say (as flow.py's TYPES)
SETTINGS = {"input": {"files": []}, "data_sheet": {"role": "object", "frames": 8, "ai": True, "what": "", "sides": ""},
            "base": {"seed": 42, "where": "server", "mode": "multidiffusion", "resolution": "1024", "steps": 25, "given": ""}, "pick_seed": {},
            "edit": {"replace": False, "width": 0.75, "sink": 0.15, "head": 0.12, "fit": "between", "prompt": "", "prompt_seed": 1, "at": ""},
            "inpaint": {"seeds": [1]}, "judge": {"ai": True, "keep_min": 85, "part_min": 70},
            "place": {"zoom": 0.25, "size": 0.8, "sides": 4, "behind": 4, "min_voxels": 5}, "check": {"ai": True}, "finish": {}, "part_views": {"way": "angles", "seed": 1}}
CHOICES = {"role": ["object", "part"], "where": ["server", "pc"], "fit": ["between", "over"]}
# Can this computer run the AI itself? Only with an NVIDIA GPU (TRELLIS.2 needs CUDA). Without one (a laptop: its
# graphics are in the processor) every run goes to the uni server as one job, whatever a page asks for; this computer
# only sends the data and shows the state. flow_page.py --gpu / --no-gpu say it instead of looking for nvidia-smi.
GPU_HERE = shutil.which("nvidia-smi") is not None
# The page runs on the uni server itself (flow_page.py --on-server, or INPAINTING_ON_SERVER=1): "the uni server" is
# this computer, so every node runs here on its GPUs, nothing goes over Sciebo, and the city says "uni server".
ON_SERVER = os.environ.get("INPAINTING_ON_SERVER") == "1"
NO_GPU = "this computer has no NVIDIA GPU: %s runs only on the uni server"
PROMPTS = {}  # {"<flow file>/<node>": {"process", "log", "prompt", "started"}}: the picture AI painting a prompt
SHOWN = core.PICTURE_TYPES + core.VIDEO_TYPES
FRESH = 60  # seconds: a "running" in flow_state.json older than this is from a run that was stopped


def url(path):
    """The page's address of a file of the project (None when it is not there or outside the project)."""
    if not path or not os.path.isfile(path):
        return None
    relative = os.path.relpath(os.path.abspath(path), core.PROJECT)
    if relative.startswith("..") or not relative.lower().endswith(SHOWN):
        return None
    return f"/file/{urllib.parse.quote(relative)}?m={int(os.path.getmtime(path))}"


def model_url(path):
    """The page's address of a model in outputs/ (the main page shows it in 3D)."""
    if not path or not os.path.isfile(path) or not path.lower().endswith(".glb"):
        return None
    relative = os.path.relpath(os.path.abspath(path), core.PROJECT).replace(os.sep, "/")
    if not relative.startswith("outputs/"):
        return None
    return f"/file/{urllib.parse.quote(relative)}?m={int(os.path.getmtime(path))}"


def short(path):
    return os.path.relpath(path, core.PROJECT) if path else None


def flow_list():
    found = []
    for name in sorted(os.listdir(FLOWS)) if os.path.isdir(FLOWS) else []:
        if name.endswith(".json"):
            try:
                one = json.load(open(os.path.join(FLOWS, name)))
                found.append({"file": name[:-5], "name": one["name"], "about": one.get("about", ""), "nodes": len(one["nodes"]),
                              "app": one.get("made_by") == "app"})
            except (ValueError, KeyError, OSError):
                pass  # not a flow file
    return found


# ---------------------------------------------------------------- what a node shows

def edit_sides(folder):
    """The edit's pictures per side, by the number of the view (make_edit.py: view_<n>.png is the model from side n;
    before/after/mask without a number are the edit's main side, _1.._3 the next sides in turning order)."""
    try:
        first = json.load(open(os.path.join(folder, "edit.json"))).get("side", 0)
    except (OSError, ValueError):
        return {}
    sides = {}
    for k in range(4):
        tail = f"_{k}" if k else ""
        names = {what: os.path.join(folder, f"{what}{tail}.png") for what in ("before", "after", "mask")}
        if os.path.isfile(names["after"]):
            sides[(first + k) % 4] = names
    return sides


def compare_edit(folder):
    """03 Edit: the model before, and the plan (the part pasted into the picture), per side."""
    sides = [{"name": f"Side {view + 1}", "before": url(names["before"]), "after": url(names["after"]), "mask": url(names["mask"])}
             for view, names in edit_sides(folder).items()]
    return {"left": "Before", "right": "Plan", "groups": [{"name": "The plan", "sides": sides}]} if sides else None


def compare_judge(folder, outputs):
    """05 Judge: the original and every candidate from the same four sides (edit_judge.py renders them that way)."""
    try:
        answer = json.load(open(os.path.join(folder, "judge.json")))
    except (OSError, ValueError):
        return None
    planned = edit_sides(answer["edit"])
    labels = {os.path.abspath(c["glb"]): c["label"] for c in outputs.get("candidates", [])}
    groups = []
    for number, entry in enumerate(answer["candidates"]):
        sides = []
        for view in list(planned) or range(4):
            names = planned.get(view, {})
            side = {"name": f"Side {view + 1}", "before": url(os.path.join(answer["edit"], f"view_{view}.png")),
                    "after": url(os.path.join(entry["renders"], f"view_{view}.png")),
                    "plan": url(names.get("after")), "mask": url(names.get("mask"))}
            measured = next((s for s in entry.get("sides", []) if s.get("view") == view), {})
            side["scores"] = {k: measured[k] for k in ("kept", "part", "colour") if k in measured}
            if side["before"] and side["after"]:
                sides.append(side)
        ai = entry.get("ai") or {}
        groups.append({"name": labels.get(os.path.abspath(entry["model"]), f"candidate {number + 1}"),
                       "best": os.path.abspath(entry["model"]) == os.path.abspath(answer["best"]), "sides": sides,
                       "scores": {k: entry.get(k) for k in ("measured", "kept", "part", "colour", "psnr", "chamfer")},
                       "ai": {k: ai.get(k) for k in ("score", "why", "defects", "model")} if ai else None})
    return {"left": "Original", "right": "Result", "groups": groups, "limits": answer.get("limits"),
            "verdict": answer.get("verdict"), "why": answer.get("why")}


def sheet_views(outputs):
    """01 Data sheet: every view with what the measuring and the local AI said about it."""
    try:
        sheet = json.load(open(outputs["data"]["path"]))
    except (OSError, ValueError, KeyError):
        return None
    lists = sheet.get("lists", {})
    views = [{"name": v["name"], "url": url(v["path"]), "side": v.get("side"), "top": v.get("top"), "colour": v.get("colour"),
              "cut_off": v.get("cut_off") or [], "base": v["name"] in lists.get("base", []), "sides": v["name"] in lists.get("sides", [])}
             for v in sheet.get("views", [])]
    return {"what": sheet.get("what"), "role": sheet.get("role"), "why": lists.get("why", []), "views": views}


def hand_of(folder, node, file):
    """03 Edit, for the edit by hand: the pictures to place the part on and what is set by hand so far."""
    try:
        info = json.load(open(os.path.join(folder, "edit.json")))
        size = json.load(open(os.path.join(folder, "views.json")))["size"]
    except (OSError, ValueError, KeyError):
        return None
    painted = os.path.join(HAND, f"{file}_{node['id']}")
    made = [{**one, "url": url(os.path.join(painted, f"prompt_{one['number']}.png")),
             "area": url(os.path.join(painted, f"prompt_{one['number']}_area.png"))} for one in prompt_list(painted)]
    job = PROMPTS.get(f"{file}/{node['id']}")
    if job:
        code = job["process"].poll()
        try:
            tail = open(job["log"], errors="replace").read()[-1200:].strip().splitlines()[-4:]
        except OSError:
            tail = []
        job = {"active": code is None, "failed": code not in (None, 0), "prompt": job["prompt"], "started": job["started"], "log": tail}
    chosen = re.fullmatch(r"prompt_(\d+)\.png", os.path.basename(node.get("after") or ""))
    sides = []
    for k in range(4):
        tail = f"_{k}" if k else ""
        view = (info["side"] + k) % 4
        sides.append({"view": view, "render": url(os.path.join(folder, f"view_{view}.png")),
                      "before": url(os.path.join(folder, f"before{tail}.png")), "after": url(os.path.join(folder, f"after{tail}.png")),
                      "mask": url(os.path.join(folder, f"mask{tail}.png")), "cut": url(os.path.join(folder, f"cut_{k}.png")),
                      "add": url(os.path.join(painted, f"add_{view}.png")), "erase": url(os.path.join(painted, f"erase_{view}.png"))})
    return {"size": size, "side": info["side"], "part": info["part_pixels"], "by_hand": bool(node.get("place") or node.get("hand") or node.get("after")),
            "replace": bool(node.get("replace", False)), "sides": sides, "folder": folder,
            "prompts": [one for one in made if one["url"]], "prompt_job": job, "after": int(chosen.group(1)) if chosen else None,
            "cubes": {k: info.get(k) for k in ("keep", "delete", "edit")}}


def sent_job(folder):
    """The name of the uni server job a node's folder was sent as (cloud.py --wait notes it)."""
    try:
        return open(os.path.join(folder, "sent_job.txt")).read().strip()
    except (OSError, TypeError):
        return None


def last_run(one):
    """flow_state.json of the last (or the running) flow.py: {node: its state}, and whether it is being written now."""
    path = os.path.join(one.folder, "flow_state.json")
    try:
        return json.load(open(path))["nodes"], time.time() - os.path.getmtime(path) < FRESH
    except (OSError, ValueError, KeyError):
        return {}, False


def describe(file, path=None):
    """One flow as the page wants it. path: the flow file when it is not in flows/ (a flow that runs on the uni server
    is described there, by run_flow.py, and the answer comes back through the cloud)."""
    one = flows.Flow(path or os.path.join(FLOWS, file + ".json"))
    one.peek(previews=False)
    saved, fresh = last_run(one)
    place, _ = one.layout()
    nodes = []
    for key, node in one.nodes.items():
        state = one.state[key]
        before = saved.get(key) or {}
        status, note = state["status"], state["note"]
        if status != "done":
            if before.get("status") in ("running", "starting") and fresh:
                status, note = before["status"], before.get("note", "")
            elif before.get("status") == "failed" and before.get("folder") in (None, state["folder"]):
                status, note = "failed", before.get("note", "")
            elif state["folder"] and os.path.isfile(os.path.join(state["folder"], "sent_job.txt")):
                job = open(os.path.join(state["folder"], "sent_job.txt")).read().strip()
                note = f"sent to the uni server as {job}, its result is not fetched yet (run the flow again: it waits for it)"
        outputs = state["outputs"] or {}
        title, _, where, _ = flows.TYPES[node["type"]]
        shown = {"id": key, "type": node["type"], "where": where, "colour": flows.COLOURS[node["type"]],
                 "label": node.get("label") or (f"{title}: {node['step']}" if "step" in node else title),
                 "column": place[key][0], "row": place[key][1], "status": status, "note": note,
                 "minutes": state.get("minutes"), "settings": one.settings(node), "inputs": flows.links(node),
                 "defaults": SETTINGS.get(node["type"], {}), "server_job": sent_job(state["folder"]),
                 "folder": short(state["folder"]), "model": short((outputs.get("model") or {}).get("glb")),
                 "model_url": model_url((outputs.get("model") or {}).get("glb")),
                 "verdict": outputs.get("verdict"), "picture": url(outputs.get("sheet")), "area_map": url(outputs.get("area_map"))}
        if state["status"] == "done":
            if node["type"] == "input":
                shown["files"] = [{"name": os.path.basename(f), "url": url(f), "video": f.lower().endswith(core.VIDEO_TYPES)}
                                  for f in outputs["pictures"]["files"]]
                shown["picture"] = next((f["url"] for f in shown["files"] if not f["video"]), None)
            elif node["type"] == "data_sheet":
                shown["sheet"] = sheet_views(outputs)
            elif node["type"] == "edit":
                shown["compare"] = compare_edit(state["folder"])
                shown["hand"] = hand_of(state["folder"], node, file)
            elif node["type"] == "judge":
                shown["compare"] = compare_judge(state["folder"], outputs)
            if len(outputs.get("models") or []) > 1:
                shown["models"] = [{"label": m["label"], "path": short(m["glb"])} for m in outputs["models"]]
        nodes.append(shown)
    running = any(n["status"] in ("running", "starting") for n in nodes)
    painting = any(key.startswith(file + "/") and job["process"].poll() is None for key, job in PROMPTS.items())
    return {"file": file, "name": one.name, "about": one.file.get("about", ""), "prompt": one.file.get("prompt", ""),
            "app": one.file.get("made_by") == "app", "running": running, "nodes": nodes, "runs": one.file.get("runs"),
            "time": time.strftime("%H:%M:%S"), "run": run_state(file), "painting": painting, "choices": CHOICES}


# ---------------------------------------------------------------- running and changing a flow

def flow_path(file):
    if file not in [f["file"] for f in flow_list()]:
        raise ValueError(f"no flow {file}")
    return os.path.join(FLOWS, file + ".json")


def run_state(file):
    """The flow.py this page started for a flow: is it running, how it ended, the end of its output."""
    run = RUNS.get(file)
    if not run:
        return None
    code = run["process"].poll()
    try:
        tail = open(run["log"], errors="replace").read()[-3000:].splitlines()[-14:]
    except OSError:
        tail = []
    return {"active": code is None, "code": code, "upto": run["upto"], "started": run["started"], "log": tail}


def project_python():
    """The project's own Python and its surroundings, as after `source activate.sh`."""
    # the environment is a Linux one (install.sh): on Windows its python can't be started, the page's own Python is used
    python = ENV_PYTHON if os.path.isfile(ENV_PYTHON) and os.name != "nt" else sys.executable
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PATH": os.path.dirname(python) + os.pathsep + os.environ.get("PATH", "")}
    env.setdefault("HF_HUB_CACHE", os.path.join(core.PROJECT, "local_ai_models"))
    env.setdefault("HF_MODULES_CACHE", os.path.join(core.PROJECT, "local_ai_models", "modules"))
    without_dead_proxy(env)
    return python, env


def without_dead_proxy(env):
    """A proxy that is set but refuses (as JupyterLab's on the uni server, 127.0.0.1:3129, sometimes) would end a run
    at the first model that is looked up, although the models are on disk. Then the run goes without it: the models
    from disk only, everything else directly (as cloud.py and cloud_worker.py do). This computer's own servers (the
    local AI, ComfyUI) are never asked through a proxy."""
    names = ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY")
    local = [a for a in (env.get("no_proxy") or env.get("NO_PROXY") or "").split(",") if a.strip()] + ["127.0.0.1", "localhost"]
    env["no_proxy"] = env["NO_PROXY"] = ",".join(dict.fromkeys(local))
    proxy = next((env[n] for n in names if env.get(n)), None)
    if not proxy:
        return
    address = urllib.parse.urlparse(proxy if "://" in proxy else "http://" + proxy)
    try:
        socket.create_connection((address.hostname, address.port or 80), timeout=3).close()
    except OSError:
        for name in names:
            env.pop(name, None)
        env.setdefault("HF_HUB_OFFLINE", "1")


# ---------------------------------------------------------------- a flow that runs completely on the uni server
# For a computer without a GPU: the flow file says "runs": "server". Run then sends the flow and its files as one job
# (cloud.py: pack_flow), the server runs every node (run_flow.py) and sends its state every 15 s; a thread here
# fetches that state and the pictures it names into outputs/server_runs/<job>/, and /api/flow answers with it.

REMOTE = {}  # {flow file: {"job", "state", "active", "message"}}


def remote_note(file):
    return os.path.join(core.PROJECT, "outputs", "server_runs", f"{file}.job")


def remote_follow(file, job):
    sys.path.insert(0, os.path.join(core.SCRIPTS, "server"))
    from cloud import Cloud
    run = REMOTE[file]
    root = os.path.join(core.PROJECT, "outputs", "server_runs", job)
    os.makedirs(os.path.join(root, "root"), exist_ok=True)
    have, cloud, looks, status = {}, None, 0, {"state": "?", "message": ""}
    while run["active"]:
        try:
            cloud = cloud or Cloud()
            if looks % 5 == 0 or run["state"] is None:  # the job's own status (queued, done, failed): every 10 s is enough
                status = cloud.read_status(job) or {"state": "?", "message": ""}
                run["message"] = f"{status['state']}: {status.get('message', '')}"[:200]
            looks += 1
            try:
                state = json.loads(cloud.request("GET", f"{cloud.folder}/{job}__live_state.json", tries=1))
            except urllib.error.HTTPError:
                state = None  # the server has not started it yet
            if state:
                for address, sent in state.get("files", {}).items():
                    relative = urllib.parse.unquote(address[len("/file/"):].split("?")[0])
                    local = os.path.realpath(os.path.join(root, "root", relative))
                    if not local.startswith(os.path.realpath(root) + os.sep):
                        continue
                    if have.get(address) != sent["stamp"] or not os.path.isfile(local):
                        os.makedirs(os.path.dirname(local), exist_ok=True)
                        cloud.download(sent["name"], local + ".part")
                        os.replace(local + ".part", local)
                        have[address] = sent["stamp"]
                if state != run["state"]:
                    run["heard"] = time.time()  # this computer's clock: the page says how fresh what it shows is
                run["state"] = state
                json.dump(state, open(os.path.join(root, "state.json"), "w"))
            if status["state"] in ("done", "failed") and (state is None or not state.get("running") or status["state"] == "failed"):
                run["active"] = False
        except Exception as e:  # noqa: BLE001 (the cloud comes and goes: look again)
            run["message"] = f"the cloud does not answer ({type(e).__name__}): trying again"
            cloud = None
        time.sleep(2)


def start_remote(file):
    path = flow_path(file)
    if (REMOTE.get(file) or {}).get("active"):
        raise ValueError("this flow is running on the uni server already")
    sys.path.insert(0, os.path.join(core.SCRIPTS, "server"))
    try:
        import cloud
        connection = cloud.Cloud()
        job = cloud.send_zip(connection, cloud.pack_flow(path))
    except SystemExit as e:  # no cloud login yet, or a file from outside pictures/uploads/
        raise ValueError(str(e)) from None
    os.makedirs(os.path.dirname(remote_note(file)), exist_ok=True)
    open(remote_note(file), "w").write(job)
    REMOTE[file] = {"job": job, "state": None, "active": True, "message": "queued: waiting for the uni server"}
    threading.Thread(target=remote_follow, args=(file, job), daemon=True).start()


def stop_remote(file):
    run = REMOTE.get(file)
    if run and run["active"]:
        sys.path.insert(0, os.path.join(core.SCRIPTS, "server"))
        from cloud import Cloud
        cloud = Cloud()
        cloud.request("PUT", f"{cloud.folder}/{run['job']}__stop", data=b"stop")


def remote_state(file):
    """What /api/flow answers for a flow that runs (or last ran) on the uni server, or None for a flow of this computer."""
    run = REMOTE.get(file)
    if run is None:  # the page was started again: the last server run of this flow, as it was fetched
        try:
            job = open(remote_note(file)).read().strip()
            saved = json.load(open(os.path.join(core.PROJECT, "outputs", "server_runs", job, "state.json")))
        except (OSError, ValueError):
            return None
        run = REMOTE[file] = {"job": job, "state": saved, "active": bool(saved.get("running")), "message": ""}
        if run["active"]:
            threading.Thread(target=remote_follow, args=(file, job), daemon=True).start()
    state = run["state"]
    if state is None:  # sent, not started there yet: this computer's own view of the flow, all of it waiting
        state = describe(file)
        for node in state["nodes"]:
            node.update(status="waiting", note="")
    prefix = f"/file/outputs/server_runs/{run['job']}/root/"

    def here(value):  # the pictures are in this computer's copy
        if isinstance(value, dict):
            return {k: here(v) for k, v in value.items()}
        if isinstance(value, list):
            return [here(v) for v in value]
        return prefix + value[len("/file/"):] if isinstance(value, str) and value.startswith("/file/") else value
    answer = here({k: v for k, v in state.items() if k != "files"})
    answer.update(file=file, running=run["active"], runs="server", run={"active": run["active"]},
                  remote={"job": run["job"], "message": run["message"], **(state.get("server") or {}),
                          "heard": round(time.time() - run["heard"], 1) if run.get("heard") else None})
    return answer


def start_run(file, upto=None):
    """flow.py for this flow."""
    path = flow_path(file)
    if json.load(open(path)).get("runs") == "server" or not GPU_HERE:  # every node on the uni server (the whole flow:
        return start_remote(file)                                       # it has no "up to here")
    if (RUNS.get(file) or {}).get("process") and RUNS[file]["process"].poll() is None:
        raise ValueError("this flow is running already")
    nodes = [n["id"] for n in json.load(open(path))["nodes"]]
    if upto is not None and upto not in nodes:
        raise ValueError(f"no node {upto}")
    python, env = project_python()
    folder = core.flow_folder(json.load(open(path))["name"])
    os.makedirs(folder, exist_ok=True)
    log = os.path.join(folder, "run.log")
    command = [python, os.path.join(core.SCRIPTS, "pipeline", "flow.py"), path] + (["--upto", upto] if upto else [])
    with open(log, "w") as out:
        out.write(f"$ {' '.join(command)}\n")
        out.flush()
        process = subprocess.Popen(command, cwd=core.PROJECT, env=env, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    RUNS[file] = {"process": process, "upto": upto, "log": log, "started": time.strftime("%H:%M:%S")}


def stop_run(file):
    if (REMOTE.get(file) or {}).get("active"):
        return stop_remote(file)
    run = RUNS.get(file)
    if run and run["process"].poll() is None:
        group = hasattr(os, "killpg")  # not on Windows: there only flow.py itself is stopped
        os.killpg(run["process"].pid, 15) if group else run["process"].terminate()  # flow.py and the step it runs
        try:
            run["process"].wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(run["process"].pid, 9) if group else run["process"].kill()
        state = os.path.join(os.path.dirname(run["log"]), "flow_state.json")  # what flow.py last wrote still says "running"
        try:
            saved = json.load(open(state))
            for node in saved["nodes"].values():
                if node["status"] in ("running", "starting"):
                    node.update(status="waiting", note="stopped")
            json.dump(saved, open(state, "w"), indent=1)
        except (OSError, ValueError, KeyError):
            pass


def save_flow(path, data):
    """A flow file as it is written by hand: one node per line."""
    lines = ["{"] + [f" {json.dumps(key)}: {json.dumps(value, ensure_ascii=False)}," for key, value in data.items() if key != "nodes"]
    lines.append(' "nodes": [')
    lines += ["  " + json.dumps(node, ensure_ascii=False) + ("," if number < len(data["nodes"]) - 1 else "")
              for number, node in enumerate(data["nodes"])]
    with open(path + ".part", "w") as f:
        f.write("\n".join(lines + [" ]", "}"]) + "\n")
    os.replace(path + ".part", path)


# ---------------------------------------------------------------- the main page: uploads become a flow

MODEL_TYPES = (".glb",)  # a 3D model the run can start from (the object's, in place of the base model TRELLIS.2 builds)


def vision_ai_here():
    """Does the local vision AI (Ollama) answer on this computer? Without it the data sheet and the judges measure alone."""
    import urllib.request
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    try:
        urllib.request.build_opener(urllib.request.ProxyHandler({})).open(host + "/api/tags", timeout=2).close()
        return True
    except Exception:  # noqa: BLE001
        return False


def save_upload(address, stream, length):
    """A picture or a video the main page sends as the request's body: into pictures/uploads/<flow>/<role>/.
    Gives back its path from the project folder (with /, as the flow files have it)."""
    asked = urllib.parse.parse_qs(urllib.parse.urlparse(address).query)
    flow, role, name = (asked.get(key, [""])[0] for key in ("flow", "role", "name"))
    if not re.fullmatch(NAME, flow) or role not in ("object", "part"):
        raise ValueError("an upload needs a project name (letters, numbers, _ and -) and a role (object or part)")
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", os.path.basename(name.replace("\\", "/"))).lstrip(".")[-80:]
    model = role == "object" and name.lower().endswith(MODEL_TYPES)  # the object's own 3D model: the run starts from it
    if not (name.lower().endswith(core.PICTURE_TYPES + core.VIDEO_TYPES) or model):
        raise ValueError(f"{name or 'this file'} is no picture or video the pipeline reads "
                         f"({', '.join(core.PICTURE_TYPES + core.VIDEO_TYPES)}, or a .glb model of the object)")
    if not 0 < length <= 2_000_000_000:
        raise ValueError("an upload is at most 2 GB")
    folder = os.path.join(UPLOADS, flow, role)
    os.makedirs(folder, exist_ok=True)
    target, left = os.path.join(folder, name), length
    with open(target + ".part", "wb") as f:
        while left:
            piece = stream.read(min(left, 1 << 20))
            if not piece:
                break
            f.write(piece)
            left -= len(piece)
    if left:
        os.remove(target + ".part")
        raise ValueError(f"the upload of {name} broke off")
    os.replace(target + ".part", target)
    return os.path.relpath(target, core.PROJECT).replace(os.sep, "/")


def create_flow(wanted):
    """The main page's Run: a flow file from what was uploaded. Every object file goes into the object's input, every
    part file into the part's. Without a part the flow ends at the base model; with one it goes on as girl_crown does:
    edit, then shape -> details -> colours, each with its judge. A part can also be only words (part_prompt, no part
    file): the edit node gets them as its prompt and the local picture AI paints the part onto the model's picture.
    The prompts are written into the flow file (the flow's "prompt", an input's "prompts"). The first prompt of the
    object's files and of the part's files is what that thing is called: its data sheet's "what", which the judge reads."""
    file = str(wanted.get("flow", ""))
    if not re.fullmatch(NAME, file):
        raise ValueError("a project name has only letters, numbers, _ and - (at most 40)")
    path = os.path.join(FLOWS, file + ".json")
    if os.path.isfile(path) and json.load(open(path)).get("made_by") != "app":
        raise ValueError(f"there is a flow {file} already that was not made on this page: take another project name")
    files, prompts, called = {"object": [], "part": []}, {"object": [], "part": []}, {}
    for item in wanted.get("items") or []:
        role = item.get("role")
        inside = os.path.realpath(os.path.join(UPLOADS, file, str(role))) + os.sep
        full = os.path.realpath(os.path.join(core.PROJECT, str(item.get("path", ""))))
        if role not in files or not full.startswith(inside) or not os.path.isfile(full):
            raise ValueError(f"{item.get('path')} was not uploaded for this project")
        files[role].append(os.path.relpath(full, core.PROJECT).replace(os.sep, "/"))
        text = " ".join(str(item.get("prompt", "")).split())
        if text:
            prompts[role].append(f"{os.path.basename(full)}: {text}")
            called.setdefault(role, text[:200])
    if not files["object"]:
        raise ValueError("one of your data must be the object: the model is built from it")
    models = [f for f in files["object"] if f.lower().endswith(MODEL_TYPES)]  # the object's own model: the base, not built
    files["object"] = [f for f in files["object"] if f not in models]
    if len(models) > 1:
        raise ValueError("one 3D model of the object at most")
    if models and not files["object"]:
        raise ValueError("with a 3D model, add one picture of it too (its front): the data sheet and the judges compare with it")
    words = " ".join(str(wanted.get("part_prompt") or "").split())[:400]
    if words and files["part"]:
        raise ValueError("a project has one part: its pictures or its words, not both")
    for role, kept in {**files, "object": files["object"] + models}.items():  # what was taken off the page leaves the upload folder too
        folder = os.path.join(UPLOADS, file, role)
        for name in os.listdir(folder) if os.path.isdir(folder) else []:
            if f"pictures/uploads/{file}/{role}/{name}" not in kept:
                os.remove(os.path.join(folder, name))
    given = wanted.get("settings") or {}
    try:
        seed, steps = int(given.get("seed", 42)), int(given.get("steps", 25))
    except (TypeError, ValueError):
        raise ValueError("seed and steps are whole numbers") from None
    resolution, where = str(given.get("resolution", "1024")), given.get("where", "server")
    if resolution not in ("512", "1024", "1536") or where not in ["all", *CHOICES["where"]] or not 1 <= steps <= 100 or seed < 0:
        raise ValueError("the settings don't fit: resolution 512, 1024 or 1536; runs on all, server or pc; steps 1 to 100")
    everything = where == "all" or not GPU_HERE  # every node on the uni server: this computer only sends the data (no GPU needed here)
    where = "server" if everything else where
    if ON_SERVER:  # this computer is the uni server: every node runs right here, nothing is sent anywhere
        everything, where = False, "pc"
    said = lambda role: {"prompts": prompts[role]} if prompts[role] else {}
    nodes = [{"id": "object", "type": "input", "label": "00 Input: the object", "files": files["object"], **said("object")}]
    if files["part"]:
        nodes.append({"id": "part", "type": "input", "label": "00 Input: the part", "files": files["part"], **said("part")})
    name = lambda role: {"what": called[role]} if role in called else {}
    nodes.append({"id": "object_sheet", "type": "data_sheet", "role": "object", **name("object"), "in": {"pictures": "object"}})
    if files["part"]:
        nodes.append({"id": "part_sheet", "type": "data_sheet", "role": "part", **name("part"), "in": {"pictures": "part"}})
    nodes.append({"id": "base", "type": "base", "seed": seed, "where": where, "resolution": resolution, "steps": steps,
                  **({"given": models[0], "label": "02 Base model: yours"} if models else {}), "in": {"data": "object_sheet"}})
    if files["part"] or words:
        at = " ".join(str(wanted.get("at") or "").split())[:120]  # where the part goes, in words ("in the right hand")
        nodes.append({"id": "edit", "type": "edit", **({"at": at} if at else {}), **({"replace": True} if wanted.get("replace") else {}),
                      "in": {"model": "base", "part": "part_sheet", "object": "object_sheet"}} if files["part"]
                     else {"id": "edit", "type": "edit", "prompt": words, "in": {"model": "base", "object": "object_sheet"}})
        if files["part"]:  # the structure of docs/FIRST-SUCCESS.md: the part as its own model, placed, details + colours only
            sheet = "part_sheet"
            if len(files["part"]) == 1 and files["part"][0].lower().endswith(core.PICTURE_TYPES):  # one picture: its back is guessed
                edit_at = next(k for k, n in enumerate(nodes) if n["id"] == "edit")  # a node comes after the ones it needs
                nodes.insert(edit_at, {"id": "part_views", "type": "part_views", "in": {"data": "part_sheet"}})
                sheet = "part_views"
                nodes[edit_at + 1]["in"]["part"] = sheet
            nodes.append({"id": "part_model", "type": "base", "label": "02 The part's own model", "seed": seed, "where": where,
                          "resolution": "1024", "steps": steps, "in": {"data": sheet}})
            nodes.append({"id": "place", "type": "place", "zoom": 0.25, "size": 0.8, "in": {"part_model": "part_model", "edit": "edit"}})
            nodes.append({"id": "check", "type": "check", "in": {"edit": "place"}})
            before, steps_, edit_ = "place", ("details", "colours"), "check"
        else:  # only words: the part grows out of the model (a crown on a head), its shape from noise
            before, steps_, edit_ = "base", ("shape", "details", "colours"), "edit"
        for step in steps_:
            nodes.append({"id": step, "type": "inpaint", "step": step, "seeds": [1], "where": where, "in": {"model": before, "edit": edit_}})
            nodes.append({"id": f"judge_{step}", "type": "judge", "step": step, "in": {"candidates": step, "edit": edit_}})
            before = f"judge_{step}"
        if files["part"]:
            nodes.append({"id": "finish", "type": "finish", "in": {"model": before, "edit": edit_}})
    if not everything and not vision_ai_here():  # no vision AI where the nodes run: the data sheets and judges measure alone
        for node in nodes:
            if node["type"] in ("data_sheet", "judge", "check"):
                node["ai"] = False
    prompt = " ".join(str(wanted.get("prompt", "")).split())
    os.makedirs(FLOWS, exist_ok=True)
    save_flow(path, {"name": file, "about": prompt or "Made on the main page.", "prompt": prompt, "made_by": "app",
                     **({"runs": "server"} if everything else {}), "nodes": nodes})


UNDERSTAND = """A user of a tool that changes 3D models wrote a message and attached data. Work out what each piece of data is for.
The message: "%s"
The data, numbered:
%s
Roles: "object" = the thing that is to be changed (the model is built from it; several pictures or a video of the same thing
are all "object"); "part" = a new thing that is to be put onto the object or to replace something of it; "unused" = neither.
With one thing only, all data are "object". If the message asks for a new part that no data shows (e.g. "give him a hat"),
there is no "part" data: put the new part's description into part_words.
Answer only with JSON, no other text:
{"roles": ["object" or "part" or "unused", one per data, in order],
 "old_part": "what of the object goes away, a few words, or empty",
 "place": "where the new part goes, e.g. in the right hand, or empty",
 "part_words": "the new part described in words, only when no data shows it, else empty",
 "say": "one short friendly sentence that tells the user what you understood"}"""


def understand(wanted):
    """The chat's words and data read by the local AI (Qwen3.6-27B, Ollama): which data is the object, which the new
    part, what goes away and where the part goes. The page shows it as a plan the user can change before anything runs."""
    import urllib.request
    if not GPU_HERE and "OLLAMA_HOST" not in os.environ:  # the page reads the plan by its simple rules then
        raise ValueError(NO_GPU % "the local AI (Qwen3.6-27B)")
    from judge import MODEL, OLLAMA
    prompt = " ".join(str(wanted.get("prompt", "")).split())[:1500]
    items = [i for i in (wanted.get("items") or []) if isinstance(i, dict)][:12]
    if not items:
        raise ValueError("no data to understand")
    lines, images = [], []
    for k, item in enumerate(items):
        thumb = str(item.get("thumb") or "")
        shown = ""
        if thumb.startswith("data:image/") and "," in thumb and len(thumb) < 600_000:
            images.append(thumb.split(",", 1)[1])
            shown = f", shown as image {len(images)}"
        lines.append(f"{k + 1}. {str(item.get('kind', 'file'))[:20]} \"{str(item.get('name', ''))[:80]}\"{shown}")
    body = {"model": MODEL, "stream": False, "format": "json", "think": False, "options": {"temperature": 0, "num_predict": 4096},
            "messages": [{"role": "user", "content": UNDERSTAND % (prompt.replace('"', "'"), "\n".join(lines)), "images": images}]}
    try:
        request = urllib.request.Request(OLLAMA + "/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"})
        answer = json.loads(json.load(urllib.request.urlopen(request, timeout=120))["message"]["content"])
    except (OSError, ValueError, KeyError) as e:
        raise ValueError(f"the local AI did not answer ({type(e).__name__})") from None
    roles = [r if r in ("object", "part", "unused") else "unused" for r in (answer.get("roles") or [])][:len(items)]
    roles += ["unused"] * (len(items) - len(roles))
    if "object" not in roles:
        roles[0] = "object"
    text = lambda key, most: " ".join(str(answer.get(key) or "").split())[:most]
    return {"roles": roles, "old_part": text("old_part", 80), "place": text("place", 120), "part_words": text("part_words", 400),
            "say": text("say", 300), "by": MODEL}


# ---------------------------------------------------------------- the bridge: the Sciebo login and the connection
# The city's bridge is the cloud folder. A click on it shows the login (never its password), saves a new one (it is
# tested first, as cloud.py setup does) and tests the way to the uni server in two steps: 1 the cloud answers with
# this login (seconds), 2 the worker on the server answers a check job (cloud_worker.py looks every 30 s).

BRIDGE = {"state": "idle", "message": "", "found": None, "job": None, "started": None}  # the last test of step 2

# The direct way: inside the uni network (on campus, or the uni VPN) the page that runs on the uni server itself
# answers. There everything runs on its GPUs with nothing in between, so a computer without a GPU opens that page
# instead of sending over Sciebo. INPAINTING_SERVER_PAGE names it; asked at most every 30 s, never through a proxy.
SERVER_PAGE = os.environ.get("INPAINTING_SERVER_PAGE", "http://nc.hcigroup.de:8777").rstrip("/")
DIRECT = {"checked": 0.0, "reachable": False, "broken": False}


def direct_server():
    """{"url", "reachable", "broken"}: does the uni server's own page answer from here, and does it work (a page started
    before a git pull that moved its files answers with errors)? None on the server itself (it has a GPU)."""
    if GPU_HERE:
        return None
    if time.time() - DIRECT["checked"] > 30:
        import urllib.request
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            opener.open(SERVER_PAGE + "/login", timeout=2).close()
            reachable, broken = True, False
        except urllib.error.HTTPError as e:  # it answered: a page with no login sends /login on, a failing one says 500
            reachable, broken = True, e.code >= 500
        except Exception:  # noqa: BLE001 (outside the uni network: nothing answers)
            reachable, broken = False, False
        DIRECT.update(checked=time.time(), reachable=reachable, broken=broken)
    return {"url": SERVER_PAGE + "/app", "reachable": DIRECT["reachable"], "broken": DIRECT["broken"]}


def cloud_module():
    sys.path.insert(0, os.path.join(core.SCRIPTS, "server"))
    import cloud
    return cloud


def cloud_login():
    """GET /api/cloud: the saved login without its password, and the last test."""
    cloud = cloud_module()
    try:
        config = json.load(open(cloud.CONFIG))
    except (OSError, ValueError):
        config = None
    login = {"user": config.get("user"), "url": config.get("url"), "folder": config.get("folder", cloud.FOLDER)} if config else None
    return {"login": login, "server": {k: v for k, v in BRIDGE.items() if k != "job"}, "gpu": GPU_HERE, "direct": direct_server()}


def save_cloud_login(wanted):
    """POST /api/cloud/login {"user", "password"}: tried first (the folder is made and listed), saved only if it works."""
    cloud = cloud_module()
    user, password = str(wanted.get("user") or "").strip(), str(wanted.get("password") or "").strip()
    if not user or not password:
        raise ValueError("the login name and the app password are both needed")
    config = {"url": cloud.DEFAULT_URL, "user": user, "password": password, "folder": cloud.FOLDER}
    try:
        old = json.load(open(cloud.CONFIG))  # another url or folder chosen with cloud.py setup stays
        config.update(url=old.get("url", config["url"]), folder=old.get("folder", config["folder"]))
    except (OSError, ValueError):
        pass
    test_cloud(cloud.Cloud(config))
    os.makedirs(os.path.dirname(cloud.CONFIG), exist_ok=True)
    with open(os.open(cloud.CONFIG, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as f:  # only its owner can read it
        json.dump(config, f)
    BRIDGE.update(state="idle", message="", found=None)


def test_cloud(connection=None):
    """Step 1: the cloud answers with this login. Gives how many files are in the jobs folder."""
    cloud = cloud_module()
    try:
        connection = connection or cloud.Cloud()
        connection.make_folder()
        return len(connection.list())
    except SystemExit as e:  # no login saved yet
        raise ValueError("No Sciebo login yet: click the bridge in the city and enter it") from None
    except urllib.error.HTTPError as e:
        raise ValueError(f"Sciebo refused the login ({e.code}): check the login name and the app password" if e.code in (401, 403)
                         else f"Sciebo answered {e.code}") from None
    except OSError as e:
        raise ValueError(f"Sciebo can't be reached ({type(e).__name__}): is this computer online?") from None


def test_server():
    """Step 2, in its own thread: a check job; the worker on the uni server answers what it has (GPUs, Blender, ...)."""
    import zipfile
    cloud = cloud_module()
    try:
        connection = cloud.Cloud()
        folder = os.path.join(core.PROJECT, "outputs", "cloud")
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, "check.zip")
        with zipfile.ZipFile(path, "w") as z:
            z.writestr("job.json", json.dumps({"kind": "check"}))
        job = BRIDGE["job"] = cloud.send_zip(connection, path)
        BRIDGE.update(state="waiting", message="waiting for the uni server to take it (it looks every 30 s)")
        while time.time() - BRIDGE["started"] < 300:
            time.sleep(5)
            status = connection.read_status(job) or {"state": "?"}
            if status["state"] == "running":
                BRIDGE.update(message="the uni server took it and looks what it has")
            if status["state"] == "failed":
                raise ValueError("the uni server answered, but the check failed: " + str(status.get("message", ""))[:200])
            if status["state"] == "done":
                local = os.path.join(core.PROJECT, "outputs", "server_runs", job, "check.zip")
                os.makedirs(os.path.dirname(local), exist_ok=True)
                connection.download(status["results"][0], local)
                with zipfile.ZipFile(local) as z:
                    found = json.loads(z.read("check.json"))
                BRIDGE.update(state="done", message=f"the uni server answered in {round(time.time() - BRIDGE['started'])} s",
                              found={"gpus": found.get("gpus", []), "code": found.get("code"), "disk_free_gb": found.get("disk_free_gb"),
                                     "parts": [{"name": p["name"], "there": p["there"]} for p in found.get("parts", [])]})
                return
        raise ValueError("no answer from the uni server in 5 minutes: is cloud_worker.py running there?")
    except (ValueError, SystemExit) as e:
        BRIDGE.update(state="failed", message=str(e))
    except Exception as e:  # noqa: BLE001 (the cloud comes and goes)
        BRIDGE.update(state="failed", message=f"the cloud stopped answering ({type(e).__name__})")


def start_server_test():
    if BRIDGE["state"] in ("sending", "waiting"):
        return
    BRIDGE.update(state="sending", message="the check goes into the cloud folder", found=None, started=time.time())
    threading.Thread(target=test_server, daemon=True).start()


def server_jobs(only=None):
    """The jobs in the cloud folder of the uni server: [{"state", "job", "time", "message"}] (cloud.py status).
    only = the names of the jobs that are asked for; all of them take over a minute (one question to the cloud each)."""
    python, env = project_python()
    if only is not None and not only:
        return {"jobs": [], "read": time.strftime("%H:%M:%S"), "text": "No node of this flow was sent to the uni server yet."}
    try:
        answer = subprocess.run([python, os.path.join(core.SCRIPTS, "server", "cloud.py"), "status", *(only or [])], cwd=core.PROJECT,
                                env=env, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        raise ValueError("the cloud did not answer within five minutes") from None
    if answer.returncode != 0:
        raise ValueError("the cloud can't be read: " + (answer.stderr or answer.stdout).strip().splitlines()[-1][:200])
    jobs = []
    for line in answer.stdout.splitlines():
        found = re.match(r"\s*(\S+)  (\S+)  (\d\d:\d\d:\d\d|)  (.*)", line)
        if found:
            jobs.append(dict(zip(("state", "job", "time", "message"), found.groups())))
            jobs[-1]["message"] = re.sub(r"\s*\[\d+ result\(s\)\]$", "", jobs[-1]["message"])
    return {"jobs": jobs, "read": time.strftime("%H:%M:%S"), "text": answer.stdout.strip() if not jobs else ""}


def set_settings(file, wanted):
    """A node's settings from the page into the flow file. Each keeps the kind of its default (number, yes/no, list of
    numbers, text); one that is the default and was not in the file stays out of it."""
    path = flow_path(file)
    data = json.load(open(path))
    node = next((n for n in data["nodes"] if n["id"] == wanted.get("node")), None)
    if node is None:
        raise ValueError("no node with this name")
    known = SETTINGS.get(node["type"], {})
    for key, value in (wanted.get("settings") or {}).items():
        if key not in known:
            raise ValueError(f"{key} can't be set on this node")
        default = known[key]
        try:
            if isinstance(default, bool):
                value = bool(value)
            elif isinstance(default, list):
                items = value if isinstance(value, list) else [v for v in re.split(r"[;,\s]+" if key != "files" else r"\s*[;\n]\s*", str(value)) if v]
                value = [str(v).strip() for v in items] if key == "files" else [int(v) for v in items]
                if not value:
                    raise ValueError
            elif isinstance(default, int):
                value = int(float(value))
            elif isinstance(default, float):
                value = float(value)
            else:
                value = str(value).strip()
        except (TypeError, ValueError):
            raise ValueError(f"{key}: {value!r} doesn't fit (it is {default!r} when nothing is set)") from None
        if key in CHOICES and value not in CHOICES[key]:
            raise ValueError(f"{key} is one of {', '.join(CHOICES[key])}")
        if key in node or value != default:
            node[key] = value
    save_flow(path, data)


def open_in_blender(model):
    """A model of outputs/ in a new Blender window (an empty scene with only this model)."""
    path = os.path.realpath(os.path.join(core.PROJECT, str(model)))
    if not path.startswith(os.path.realpath(os.path.join(core.PROJECT, "outputs")) + os.sep) or not path.endswith(".glb") \
            or not os.path.isfile(path):
        raise ValueError("no such model")
    blender = flows.BLENDER
    if not os.path.isfile(blender):  # not where flow.py expects it: on the PATH, or Windows' usual place (the newest)
        import glob
        usual = sorted(glob.glob(os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "Blender Foundation", "*", "blender.exe")))
        blender = shutil.which("blender") or (usual[-1] if usual else "")
    if not blender or not os.path.isfile(blender):
        raise ValueError(f"Blender is not at {flows.BLENDER} (set BLENDER before starting the page)")
    script = ("import bpy\nfor o in list(bpy.data.objects):\n    bpy.data.objects.remove(o)\n"
              f"bpy.ops.import_scene.gltf(filepath={path!r})\n"
              "for area in (bpy.context.screen.areas if bpy.context.screen else []):\n    if area.type == 'VIEW_3D':\n"
              "        area.spaces[0].shading.type = 'MATERIAL'\n")
    subprocess.Popen([blender, "--python-expr", script], cwd=core.PROJECT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     start_new_session=True)


def edit_node(file, wanted):
    """The flow file's data, the edit node that is meant, and its folder of things made by hand."""
    data = json.load(open(flow_path(file)))
    node = next((n for n in data["nodes"] if n["id"] == wanted.get("node") and n["type"] == "edit"), None)
    if node is None:
        raise ValueError("no edit node with this name")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", file + node["id"]):
        raise ValueError("the flow's and the node's name may only have letters, numbers, _ and -")
    return data, node, os.path.join(HAND, f"{file}_{node['id']}")


def save_picture(target, picture):
    """A PNG the page sent as a data URL."""
    raw = base64.b64decode(str(picture).split(",", 1)[-1])
    if not raw.startswith(b"\x89PNG") or len(raw) > 8_000_000:
        raise ValueError(f"{os.path.basename(target)} is no PNG picture")
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, "wb") as f:
        f.write(raw)


def save_layers(folder, layers):
    """The painted corrections of the 2D mask: {"add_0": a PNG as data URL, or None = nothing painted there}."""
    for name, picture in (layers or {}).items():
        if not re.fullmatch(r"(add|erase)_[0-3]", name):
            raise ValueError(f"no such layer {name}")
        target = os.path.join(folder, name + ".png")
        if picture is not None:
            save_picture(target, picture)
        elif os.path.isfile(target):
            os.remove(target)


def prompt_list(folder):
    """The pictures made from prompts so far: [{"number", "prompt", "seed", "reference", "time"}]."""
    try:
        return json.load(open(os.path.join(folder, "prompts.json")))
    except (OSError, ValueError):
        return []


def start_prompt(file, wanted):
    """The local picture AI paints a prompt into the painted area of the edit's main side (about half a minute)."""
    if not GPU_HERE:
        raise ValueError(NO_GPU % "the picture AI" + " (describe the new part in words in the plan: it is painted there)")
    _, node, folder = edit_node(file, wanted)
    key = f"{file}/{node['id']}"
    if key in PROMPTS and PROMPTS[key]["process"].poll() is None:
        raise ValueError("a picture is being made already")
    prompt = " ".join(str(wanted.get("prompt", "")).split())
    if not prompt:
        raise ValueError("write what should be there first")
    edit = os.path.realpath(str(wanted.get("folder", "")))  # the edit's result folder the page shows (its BEFORE picture)
    if not edit.startswith(os.path.realpath(core.OUTPUTS) + os.sep) or not os.path.isfile(os.path.join(edit, "edit.json")):
        raise ValueError("run the edit node once first: its picture of the model is needed")
    side = json.load(open(os.path.join(edit, "edit.json")))["side"]
    if not wanted.get("area"):
        raise ValueError("paint the area on the main side first (red): the prompt is painted into it")
    made = prompt_list(folder)
    number = 1 + max([one["number"] for one in made], default=0)
    out = os.path.join(folder, f"prompt_{number}.png")
    area = os.path.join(folder, f"prompt_{number}_area.png")  # its own copy: the flow's painted mask (add_<view>.png) only
    save_picture(area, wanted["area"])                        # changes with Apply, or the edit node would count as not made
    python, env = project_python()
    command = [python, os.path.join(core.SCRIPTS, "inpainting", "picture_from_prompt.py"), os.path.join(edit, "before.png"), area, out,
               "--prompt", prompt, "--seed", str(int(wanted.get("seed") or 1))]
    sides = os.path.join(edit, "part_sides")
    pictures = sorted(os.listdir(sides)) if os.path.isdir(sides) else []
    if wanted.get("reference") and pictures:
        command += ["--reference", os.path.join(sides, pictures[0])]
    log = os.path.join(folder, "prompt.log")
    with open(log, "w") as output:
        process = subprocess.Popen(command, cwd=core.PROJECT, env=env, stdout=output, stderr=subprocess.STDOUT)
    PROMPTS[key] = {"process": process, "log": log, "prompt": prompt, "started": time.strftime("%H:%M:%S")}

    def note():  # when it is done: into the list of this node's prompts
        if process.wait() == 0 and os.path.isfile(out):
            entry = {"number": number, "prompt": prompt, "seed": int(wanted.get("seed") or 1), "side": side,
                     "reference": bool(wanted.get("reference") and pictures), "time": time.strftime("%Y-%m-%d %H:%M")}
            json.dump(prompt_list(folder) + [entry], open(os.path.join(folder, "prompts.json"), "w"), indent=1)
    threading.Thread(target=note, daemon=True).start()


def edit_by_hand(file, wanted):
    """Writes an edit by hand into the flow: the place of the part and `replace` into the node's settings, the painted
    corrections of the 2D mask into flows/hand/<flow>_<node>/. place = None: back to the measured place.
    after = the number of a picture made from a prompt: it is the AFTER picture then, no part picture is pasted."""
    path = flow_path(file)
    data, node, folder = edit_node(file, wanted)
    save_layers(folder, wanted.get("layers"))
    place, after = wanted.get("place"), wanted.get("after")
    if after is not None:
        picture = os.path.join(folder, f"prompt_{int(after)}.png")
        side = next((one["side"] for one in prompt_list(folder) if one["number"] == int(after)), None)
        if side is None or not os.path.isfile(picture):
            raise ValueError("this prompt's picture is not there any more")
        if not os.path.isfile(os.path.join(folder, f"add_{side}.png")):  # nothing painted since: the area the prompt was painted into
            if not os.path.isfile(os.path.join(folder, f"prompt_{int(after)}_area.png")):
                raise ValueError("paint the area on the main side (red): what the prompt made is taken from inside it")
            shutil.copyfile(os.path.join(folder, f"prompt_{int(after)}_area.png"), os.path.join(folder, f"add_{side}.png"))
        node["after"], node["side"] = os.path.relpath(picture, core.PROJECT), side
        node.pop("place", None)
    elif place:
        node["place"], node["side"] = [int(round(float(v))) for v in place[:3]], int(wanted["side"])
        node.pop("after", None)
    else:
        for key in ("place", "side", "after"):
            node.pop(key, None)
    if "replace" in wanted:
        node["replace"] = bool(wanted["replace"])
    if os.path.isdir(folder) and any(f.startswith(("add_", "erase_")) and f.endswith(".png") for f in os.listdir(folder)):
        node["hand"] = os.path.relpath(folder, core.PROJECT)
    else:
        node.pop("hand", None)
    save_flow(path, data)
    return node


# ---------------------------------------------------------------- the server

class Page(BaseHTTPRequestHandler):
    seen, lock = {}, threading.Lock()  # {flow file: (time, its description)}: several tabs don't read the folders twice
    accounts = False  # --accounts: nothing without a login

    def user(self):
        """The account that is logged in (its cookie), or None."""
        for part in self.headers.get("Cookie", "").split(";"):
            name, _, value = part.strip().partition("=")
            if name == COOKIE:
                return accounts.who(value)
        return None

    def cookie(self, token, seconds):
        safe = "; Secure" if self.headers.get("X-Forwarded-Proto") == "https" else ""  # behind the web server's HTTPS
        return ("Set-Cookie", f"{COOKIE}={token}; Path=/; Max-Age={seconds}; HttpOnly; SameSite=Strict{safe}")

    def log_message(self, *args):  # quiet: a page asks every few seconds
        pass

    def send(self, body, kind, code=200, extra=()):
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        for name, value in extra:
            self.send_header(name, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def answer(self, data, code=200):
        self.send(json.dumps(data, default=str).encode(), "application/json", code, [("Cache-Control", "no-store")])

    def do_GET(self):
        address = urllib.parse.urlparse(self.path)
        try:
            if self.accounts:
                name = self.user()
                if address.path == "/login":
                    if name:
                        return self.send(b"", "text/plain", 302, [("Location", "/app")])
                    return self.send(open(LOGIN, "rb").read(), "text/html; charset=utf-8", extra=[("Cache-Control", "no-store")])
                if not name and address.path in ("/", "/app", "/old/app"):
                    return self.send(b"", "text/plain", 302, [("Location", "/login"), ("Cache-Control", "no-store")])
                if not name:
                    return self.answer({"error": "log in first"}, 401)
            if address.path == "/api/me":  # who is logged in; without --accounts nobody is asked
                return self.answer({"accounts": self.accounts, "name": name if self.accounts else None, "gpu": GPU_HERE,
                                    "on_server": ON_SERVER})
            if address.path == "/":
                return self.send(open(PAGE, "rb").read(), "text/html; charset=utf-8", extra=[("Cache-Control", "no-store")])
            if address.path in ("/app", "/old/app"):  # the main page, and the one before it
                return self.send(open(APP if address.path == "/app" else OLD_APP, "rb").read(), "text/html; charset=utf-8",
                                 extra=[("Cache-Control", "no-store")])
            if address.path == "/api/cloud":
                return self.answer(cloud_login())
            if address.path == "/api/flows":
                return self.answer(flow_list())
            if address.path == "/api/server":
                try:
                    asked = urllib.parse.parse_qs(address.query)
                    if "all" in asked:
                        return self.answer(server_jobs())
                    name = asked.get("flow", [""])[0]
                    if name not in [f["file"] for f in flow_list()]:
                        return self.answer({"error": f"no flow {name}"}, 404)
                    return self.answer(server_jobs([n["server_job"] for n in describe(name)["nodes"] if n["server_job"]]))
                except ValueError as e:
                    return self.answer({"error": str(e)}, 502)
            if address.path == "/api/flow":
                name = urllib.parse.parse_qs(address.query).get("name", [""])[0]
                if name not in [f["file"] for f in flow_list()]:
                    return self.answer({"error": f"no flow {name}"}, 404)
                with self.lock:
                    if name not in self.seen or time.time() - self.seen[name][0] > 0.8:  # the page asks every second
                        self.seen[name] = (time.time(), remote_state(name) or describe(name))
                    return self.answer(self.seen[name][1])
            if address.path.startswith("/file/"):
                return self.file(urllib.parse.unquote(address.path[len("/file/"):]))
            self.answer({"error": "not here"}, 404)
        except (BrokenPipeError, ConnectionResetError):
            pass  # the page was closed while it loaded
        except Exception as e:  # noqa: BLE001 (shown on the page)
            self.answer({"error": f"{type(e).__name__}: {e}"}, 500)

    do_HEAD = do_GET

    def do_POST(self):
        """Only this page may ask: its own address, and a header a foreign web page can't send without being asked."""
        try:
            host = self.headers.get("Host", "")
            origin = self.headers.get("Origin")
            local = host.split(":")[0] in ("127.0.0.1", "localhost")  # with accounts the page has other addresses too
            if self.headers.get("X-Flow-Page") != "1" or (origin and origin not in (f"http://{host}", f"https://{host}")) \
                    or not (local or self.accounts):
                return self.answer({"error": "only the flow page itself may do this"}, 403)
            length = int(self.headers.get("Content-Length") or 0)
            if self.accounts and self.path in ("/api/login", "/api/logout"):
                wanted = json.loads(self.rfile.read(min(length, 10_000)) or b"{}")
                if self.path == "/api/logout":
                    accounts.log_out(next((p.strip()[len(COOKIE) + 1:] for p in self.headers.get("Cookie", "").split(";")
                                           if p.strip().startswith(COOKIE + "=")), ""))
                    return self.send(b'{"ok": true}', "application/json", extra=[self.cookie("", 0)])
                try:
                    token = accounts.log_in(wanted.get("name"), wanted.get("password"))
                except ValueError as e:
                    return self.answer({"error": str(e)}, 401)
                return self.send(b'{"ok": true}', "application/json", extra=[self.cookie(token, accounts.KEPT), ("Cache-Control", "no-store")])
            if self.accounts and not self.user():
                return self.answer({"error": "log in first"}, 401)
            if self.path.startswith("/api/upload?"):  # the body is the file itself, not JSON
                try:
                    return self.answer({"ok": True, "path": save_upload(self.path, self.rfile, length)})
                except ValueError:  # refused before it was read: the body is read away, or the browser never sees why
                    left = min(length, 200_000_000)
                    while left > 0:
                        piece = self.rfile.read(min(left, 1 << 20))
                        if not piece:
                            break
                        left -= len(piece)
                    raise
            if length > 80_000_000:
                return self.answer({"error": "too much"}, 413)
            wanted = json.loads(self.rfile.read(length) or b"{}")
            file = wanted.get("flow", "")
            if self.path == "/api/understand":
                return self.answer({"ok": True, **understand(wanted)})
            if self.path.startswith("/api/cloud/"):
                if self.accounts:  # the login is the one of the computer the page runs on, not of whoever is logged in
                    raise ValueError("not on a shared page: the cloud login is set on the computer the page runs on")
                if self.path == "/api/cloud/login":
                    save_cloud_login(wanted)
                    return self.answer({"ok": True, **cloud_login()})
                if self.path == "/api/cloud/test":
                    files = test_cloud()
                    if wanted.get("server", True):
                        start_server_test()
                    return self.answer({"ok": True, "cloud": f"Sciebo answers: the folder has {files} files", **cloud_login()})
            if self.path == "/api/blender":
                if self.accounts:
                    raise ValueError("not on a shared page: it would open Blender on the computer the page runs on")
                open_in_blender(wanted.get("model"))
                return self.answer({"ok": True})
            if self.path == "/api/settings":
                if (run_state(file) or {}).get("active"):
                    raise ValueError("this flow is running, stop it first")
                set_settings(file, wanted)
            elif self.path == "/api/app/create":
                REMOTE.pop(file, None) if not (REMOTE.get(file) or {}).get("active") else None  # a new plan: not the last server run
                if os.path.isfile(remote_note(file)) and file not in REMOTE:
                    os.remove(remote_note(file))
                if (run_state(file) or {}).get("active") or (REMOTE.get(file) or {}).get("active"):
                    raise ValueError("this project is running, stop it first")
                create_flow(wanted)
            elif self.path == "/api/run":
                start_run(file, wanted.get("upto"))
            elif self.path == "/api/stop":
                stop_run(file)
            elif self.path == "/api/prompt":
                start_prompt(file, wanted)
            elif self.path == "/api/edit":
                if (run_state(file) or {}).get("active"):
                    raise ValueError("this flow is running, stop it first")
                edit_by_hand(file, wanted)
                if wanted.get("run"):
                    start_run(file, wanted["node"])
            else:
                return self.answer({"error": "not here"}, 404)
            with self.lock:
                self.seen.pop(file, None)
            self.answer({"ok": True})
        except (ValueError, KeyError, OSError) as e:
            self.answer({"error": str(e)}, 400)

    def file(self, relative):
        """A picture or a video out of outputs/ or pictures/; a video also in pieces (the browser asks for a Range)."""
        # the path as written (.. taken out, links not followed): a folder the user linked into outputs/ is served too
        # (Nov 19: /home was full, the runs' work folders were moved to /tmp and linked back, and the page showed none
        # of their pictures). Only pictures, videos and models under outputs/, pictures/ and flows/, as before
        path = os.path.normpath(os.path.join(os.path.abspath(core.PROJECT), relative))
        inside = any(path.startswith(os.path.join(os.path.abspath(core.PROJECT), top) + os.sep) for top in OPEN)
        model = path.lower().endswith(".glb") and path.startswith(os.path.join(os.path.abspath(core.PROJECT), "outputs") + os.sep)
        if not inside or not (path.lower().endswith(SHOWN) or model) or not os.path.isfile(path):
            return self.answer({"error": "no such picture"}, 404)
        size = os.path.getsize(path)
        start, end, code = 0, size - 1, 200
        wanted = self.headers.get("Range", "")
        if wanted.startswith("bytes="):
            try:
                first, _, last = wanted[6:].split(",")[0].partition("-")
                start, end = (int(first), int(last) if last else size - 1) if first else (size - int(last), size - 1)
                start, end, code = max(0, start), min(end, size - 1), 206
            except ValueError:
                pass
        with open(path, "rb") as f:
            f.seek(start)
            body = f.read(end - start + 1)
        extra = [("Cache-Control", "max-age=31536000, immutable"), ("Accept-Ranges", "bytes")]  # the address has the file's time
        if code == 206:
            extra.append(("Content-Range", f"bytes {start}-{end}/{size}"))
        self.send(body, "model/gltf-binary" if model else mimetypes.guess_type(path)[0] or "application/octet-stream", code, extra)


def main():
    parser = argparse.ArgumentParser(description="The web page of the node flows (this PC only, or with --accounts for others)")
    parser.add_argument("--port", type=int, default=8777)
    parser.add_argument("--open", action="store_true", help="open the page in the browser")
    parser.add_argument("--accounts", action="store_true", help="a login first; the accounts are made with accounts.py")
    parser.add_argument("--host", default="127.0.0.1", help="the address to listen on; another one than this PC's own "
                        "(0.0.0.0 = everybody who reaches this computer) only with --accounts")
    gpu = parser.add_mutually_exclusive_group()
    gpu.add_argument("--gpu", dest="gpu", action="store_true", default=None, help="this computer runs the AI itself (default: if nvidia-smi is there)")
    gpu.add_argument("--no-gpu", dest="gpu", action="store_false", help="every run on the uni server, nothing of the AI here")
    parser.add_argument("--on-server", action="store_true", help="this computer IS the uni server: every node runs here")
    args = parser.parse_args()
    global GPU_HERE, ON_SERVER
    GPU_HERE = GPU_HERE if args.gpu is None else args.gpu
    ON_SERVER = ON_SERVER or args.on_server
    if ON_SERVER and not GPU_HERE:
        parser.error("--on-server needs the server's GPUs (nvidia-smi was not found)")
    if args.host not in ("127.0.0.1", "localhost") and not args.accounts:
        parser.error("--host for other computers needs --accounts: without a login everybody there could run and change everything")
    if args.accounts and not accounts.read():
        parser.error("no accounts yet: python scripts/pipeline/accounts.py add <name>")
    Page.accounts = args.accounts
    server = ThreadingHTTPServer((args.host, args.port), Page)
    address = f"http://127.0.0.1:{args.port}"
    if ON_SERVER:
        print("On the uni server: every node of a run works here, on its GPUs.", flush=True)
    elif not GPU_HERE:
        print("No NVIDIA GPU here: every run goes to the uni server (the Sciebo login: click the bridge in the city).", flush=True)
    print(f"The flow page: {address}   (Ctrl+C stops it)" + (f"\nwith accounts ({len(accounts.read())}), listening on {args.host}" if args.accounts else ""), flush=True)
    if args.open:
        webbrowser.open(address)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
