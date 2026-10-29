"""Send several generate jobs to the uni server through the cloud, a few at a time, and fetch each result.

For trying settings (resolution, steps, guidance, faces ...) on the same pictures without sitting next
to it. By default one job at a time, so only one GPU of the shared server is used.

    python3 scripts/server/sweep.py pictures/dasle/view_*.png --try "--resolution 1536 --max-tokens 49152" \
        --try "--steps 50" --try "--guidance 10" --parallel 1

Every --try is added to the common options (--common, default "--resolution 1024 --steps 25").
Results land in outputs/cloud/ as usual; a summary line per job is printed when it is done.
"""
import argparse
import os
import re
import shlex
import subprocess
import sys
import time

CLOUD = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "cloud.py")]


def state(job):
    """The job's status line from cloud.py status ("" while it isn't listed yet)."""
    out = subprocess.run(CLOUD + ["status"], capture_output=True, text=True).stdout
    return next((line.strip() for line in out.splitlines() if f" {job} " in f" {line.strip()} "), "")


def main():
    parser = argparse.ArgumentParser(description="Try several settings on the uni server, a few jobs at a time")
    parser.add_argument("pictures", nargs="+")
    parser.add_argument("--try", dest="tries", action="append", required=True, help="options for one job")
    parser.add_argument("--common", default="--resolution 1024 --steps 25", help="options for every job")
    parser.add_argument("--parallel", type=int, default=1, help="jobs running at the same time (default 1)")
    args = parser.parse_args()

    waiting = [shlex.split(args.common) + shlex.split(t) for t in args.tries]
    running = {}
    while waiting or running:
        while waiting and len(running) < args.parallel:
            options = waiting.pop(0)
            out = subprocess.run(CLOUD + ["generate", *args.pictures, *options], capture_output=True, text=True).stdout
            job = re.search(r"Sent as (\S+)", out)
            if not job:
                print(f"[{time.strftime('%H:%M')}] not sent ({' '.join(options)}): {out.strip()[-200:]}", flush=True)
                continue
            running[job.group(1)] = options
            print(f"[{time.strftime('%H:%M')}] sent {job.group(1)}: {' '.join(options)}", flush=True)
        time.sleep(60)
        for job in list(running):
            line = state(job)
            if line.startswith(("done", "failed")):
                if line.startswith("done"):
                    subprocess.run(CLOUD + ["fetch", job], capture_output=True)
                print(f"[{time.strftime('%H:%M')}] {line[:160]} | {' '.join(running.pop(job))}", flush=True)


if __name__ == "__main__":
    main()
