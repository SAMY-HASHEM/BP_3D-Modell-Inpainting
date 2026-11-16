"""On the uni server: one whole flow, every node of it here. For a computer without a GPU: it only sends the data
and shows the state (cloud.py flow, or the web page with "everything on the uni server").

The job's folder has flow.json and files/ (the flow's input files, at their paths from the project folder). This
  1 puts the files where the flow file names them (only under pictures/uploads/)
  2 lets every node run on this computer (a base model and an inpainting are not sent anywhere: they are here)
  3 starts flow.py, and while it runs sends the state to the cloud every 15 s: <job>__live_state.json is what the web
    page shows (flow_page.describe: every node's state, note, minutes, pictures), and every picture it names as
    <job>__live_<hash>. The last model goes up when the flow is done
  4 stops flow.py when <job>__stop appears in the cloud (the page's Stop)

    python scripts/server/run_flow.py outputs/cloud_jobs/<job> --live <job>
"""
import _paths  # noqa: F401  (the scripts of the other folders)

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import zipfile

PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder")
    parser.add_argument("--live", required=True, help="the job's name in the cloud")
    args = parser.parse_args()
    import flow_page
    from cloud import Cloud
    cloud, job = Cloud(), args.live
    flow = json.load(open(os.path.join(args.folder, "flow.json")))
    name = str(flow.get("name", ""))
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", name):
        raise SystemExit("a flow's name has only letters, numbers, _ and -")
    files = os.path.join(args.folder, "files")
    for root, _, names in os.walk(files):  # 1
        for one in names:
            relative = os.path.relpath(os.path.join(root, one), files).replace(os.sep, "/")
            target = os.path.realpath(os.path.join(PROJECT, relative))
            if not relative.startswith("pictures/uploads/") or not target.startswith(os.path.realpath(os.path.join(PROJECT, "pictures", "uploads")) + os.sep):
                raise SystemExit(f"a flow's files go under pictures/uploads/, not {relative}")
            os.makedirs(os.path.dirname(target), exist_ok=True)
            shutil.copy(os.path.join(root, one), target)
    notes = []
    try:
        from judge import OLLAMA
        urllib.request.urlopen(OLLAMA + "/api/tags", timeout=5)
        vision = True
    except Exception:  # noqa: BLE001
        vision = False
        notes.append("no vision AI (Ollama) on the server: the data sheet takes every picture as a front view, the judges only measure")
    for node in flow["nodes"]:  # 2
        if node["type"] in ("base", "inpaint"):
            node["where"] = "pc"
        if node["type"] == "data_sheet" and not vision:
            node["ai"] = False
    flow.pop("runs", None)
    path = os.path.join(args.folder, name + ".json")
    json.dump(flow, open(path, "w"), indent=1)

    process = subprocess.Popen([sys.executable, "-u", os.path.join(PROJECT, "scripts", "pipeline", "flow.py"), path], cwd=PROJECT,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace", start_new_session=True)
    last = [""]

    def pump():  # flow.py's lines go on to the worker (its newest one is the job's status line)
        for line in process.stdout:
            print(line.rstrip(), flush=True)
            last[0] = line.strip() or last[0]
    threading.Thread(target=pump, daemon=True).start()

    sent, stopped = {}, False

    def report(final=False):
        state = flow_page.describe(name, path)
        state["running"] = not final
        if final:  # what flow.py last wrote may still say "running": nothing runs any more
            for node in state["nodes"]:
                if node["status"] in ("running", "starting"):
                    node.update(status="waiting", note="stopped" if stopped else "ended before it was done")
        urls = set()

        def walk(value):
            if isinstance(value, dict):
                for v in value.values():
                    walk(v)
            elif isinstance(value, list):
                for v in value:
                    walk(v)
            elif isinstance(value, str) and value.startswith("/file/"):
                urls.add(value)
        walk(state)
        models = [n["model_url"] for n in state["nodes"] if n.get("model_url")]
        state["files"] = {}
        for address in sorted(urls):
            local = os.path.join(PROJECT, urllib.request.url2pathname(address[len("/file/"):].split("?")[0]))
            if not os.path.isfile(local):
                continue
            if local.lower().endswith(".glb") and not (final and models and address == models[-1]):
                continue  # models are big: only the last one, when the flow is done
            stamp = f"{os.path.getmtime(local):.0f}-{os.path.getsize(local)}"
            key = hashlib.sha1(address.encode()).hexdigest()[:16] + os.path.splitext(local)[1].lower()
            if sent.get(address) != stamp:
                try:
                    cloud.upload(local, f"{job}__live_{key}")
                    sent[address] = stamp
                except Exception as e:  # noqa: BLE001 (the next round tries again)
                    print(f"    not sent: {os.path.basename(local)} ({type(e).__name__})", flush=True)
                    continue
            state["files"][address] = {"name": f"{job}__live_{key}", "stamp": stamp}
        state["server"] = {"job": job, "notes": notes, "line": last[0][:200], "time": time.strftime("%H:%M:%S"), "stopped": stopped}
        local_state = os.path.join(args.folder, "state.json")
        json.dump(state, open(local_state, "w"))
        cloud.upload(local_state, f"{job}__live_state.json")
        return state

    while process.poll() is None:  # 3, 4
        try:
            report()
            try:
                cloud.request("GET", f"{cloud.folder}/{job}__stop", tries=1)
                stopped = True
                print("Stop was asked for on the page: ending the flow", flush=True)
                os.killpg(process.pid, 15)  # flow.py and the step it runs
            except urllib.error.HTTPError:
                pass
        except Exception as e:  # noqa: BLE001 (the cloud was away for a moment: the flow goes on)
            print(f"    state not sent: {type(e).__name__}: {e}", flush=True)
        for _ in range(15):
            if process.poll() is not None:
                break
            time.sleep(1)
    time.sleep(1)
    for attempt in range(5):  # the last state must arrive: it says how the flow ended
        try:
            report(final=True)
            break
        except Exception as e:  # noqa: BLE001
            print(f"    last state not sent ({type(e).__name__}), once more", flush=True)
            time.sleep(10)
    result = os.path.join(args.folder, "flow_result.zip")
    with zipfile.ZipFile(result, "w") as z:
        z.write(os.path.join(args.folder, "state.json"), "state.json")
    print(f"Result: {result}", flush=True)


if __name__ == "__main__":
    main()
