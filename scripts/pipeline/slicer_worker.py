"""The Slicer's worker: does what the page asks for, one job at a time on one GPU.

It looks every few seconds into outputs/slicer/*/: a plan_request.json is planned (slicer_plan.py: Qwen3-VL looks
at the pictures, the parts and prompts follow); the chat shows the result by itself. The pictures and the 3D models
(parts/<id>/request.json) come next; until then they wait.

    python scripts/pipeline/slicer_worker.py          # keeps running; Ctrl+C stops it
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(os.path.dirname(HERE))
ROOT = os.path.join(PROJECT, "outputs", "slicer")
sys.path.insert(0, HERE)
from slicer_page import slicer_says  # noqa: E402


def plan(name):
    folder = os.path.join(ROOT, name)
    print(time.strftime("%H:%M:%S"), "planning", name, flush=True)
    note = json.load(open(os.path.join(folder, "plan_request.json"))).get("note", "")
    run = subprocess.run([sys.executable, os.path.join(HERE, "slicer_plan.py"), folder, "--note", note],
                         capture_output=True, text=True, cwd=PROJECT)
    open(os.path.join(folder, "plan_log.txt"), "w").write(run.stdout + run.stderr)
    os.remove(os.path.join(folder, "plan_request.json"))
    if run.returncode != 0:
        last = (run.stderr.strip().splitlines() or ["?"])[-1]
        slicer_says(name, f"I couldn't plan the parts: {last[:300]}\nPlease tell me in the chat; the log is plan_log.txt.")
    print(time.strftime("%H:%M:%S"), "done" if run.returncode == 0 else "FAILED", name, flush=True)


def main():
    print("The Slicer's worker: watching", ROOT, flush=True)
    while True:
        for name in sorted(os.listdir(ROOT)) if os.path.isdir(ROOT) else []:
            if os.path.isfile(os.path.join(ROOT, name, "plan_request.json")):
                try:
                    plan(name)
                except Exception as e:  # one job going wrong (a project deleted meanwhile ...) never stops the worker
                    print(time.strftime("%H:%M:%S"), "ERROR", name, repr(e), flush=True)
                    try:
                        os.remove(os.path.join(ROOT, name, "plan_request.json"))
                    except OSError:
                        pass
        time.sleep(3)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
