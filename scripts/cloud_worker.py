"""On the uni GPU server: run the jobs that arrive in the Sciebo cloud folder, one after another.

Every 30 seconds it looks into the cloud folder inpainting-jobs/. A new job (<job>.zip with status
"queued") is downloaded and run on the GPU with the least memory in use; while it runs, its last
progress line is written to <job>.status, and the finished model(s) are uploaded as <job>__<file>.glb.
Only our own scripts run, with checked settings: run_job.py for inpainting jobs from Blender,
generate.py (one picture) or multi_image.py (several) for picture -> model jobs.

A job can ask for the newest code of a branch ("code": "main" in its job.json, from cloud.py --code):
the worker then brings this checkout to that branch's newest commit from the project's public GitHub
repository (fast-forward only; code never comes from the cloud folder), and restarts itself first if
its own file changed. So new code can be tested on the server without logging in there.

Usage (TRELLIS.2 environment, on the server, after python3 scripts/cloud.py setup there):
    source activate.sh
    nohup python -u scripts/cloud_worker.py > outputs/cloud_worker.log 2>&1 &     # keeps running
    tail -f outputs/cloud_worker.log                                               # watch it
    python scripts/cloud_worker.py --once                                          # one round only
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
import traceback
import zipfile

from cloud import PROJECT_DIR, Cloud
from run_job import freest_gpu

WORK = os.path.join(PROJECT_DIR, "outputs", "cloud_jobs")
CODE_URL = "https://github.com/SAMY-HASHEM/BP_3D-Modell-Inpainting.git"  # public: no login needed
# the settings a picture -> model job may use, with their checks, and which script knows them
CHECKS = {
    "resolution": lambda v: str(v) if str(v) in ("512", "1024", "1536") else None,
    "steps": lambda v: int(v) if 1 <= int(v) <= 100 else None,
    "seed": lambda v: int(v),
    "guidance": lambda v: float(v) if 0 <= float(v) <= 20 else None,
    "texture_guidance": lambda v: float(v) if 0 <= float(v) <= 20 else None,
    "max_tokens": lambda v: int(v) if 1000 <= int(v) <= 200000 else None,
    "faces": lambda v: int(v) if 10_000 <= int(v) <= 8_000_000 else None,
    "mirror": lambda v: bool(v),
    "mode": lambda v: v if v in ("multidiffusion", "stochastic") else None,
}
KNOWS = {
    "scripts/generate.py": {"resolution", "steps", "seed", "guidance", "texture_guidance", "max_tokens", "faces"},
    "scripts/multi_image.py": {"resolution", "steps", "seed", "guidance", "max_tokens", "faces", "mirror", "mode"},
}


def command_for(job, folder, zip_path, gpu):
    """The command that runs this job, built only from checked values."""
    if job.get("kind", "inpaint") == "inpaint":
        return [sys.executable, "-u", "scripts/run_job.py", zip_path, "--gpu", str(gpu)]
    if job["kind"] != "generate":
        raise ValueError(f"unknown kind of job: {job['kind']}")
    pictures = [os.path.join(folder, os.path.basename(p)) for p in job["pictures"]]
    for p in pictures:
        if not os.path.isfile(p):
            raise ValueError(f"picture missing in the job: {os.path.basename(p)}")
    script = "scripts/generate.py" if len(pictures) == 1 else "scripts/multi_image.py"
    command = [sys.executable, "-u", script, *pictures]
    for key, value in job.get("options", {}).items():
        if key not in CHECKS:
            raise ValueError(f"unknown setting: {key}")
        value = CHECKS[key](value)  # every value is checked, also ones this script won't use
        if value is None:
            raise ValueError(f"setting out of range: {key}={job['options'][key]}")
        if key not in KNOWS[script]:
            continue  # e.g. --mirror with one picture: generate.py doesn't have it
        flag = "--" + key.replace("_", "-")
        command += [flag] if value is True else [] if value is False else [flag, str(value)]
    return command


def use_code(branch, project=PROJECT_DIR):
    """Brings the checkout to the newest commit of `branch` on GitHub (fast-forward only). Returns that
    commit ("<hash> <subject>") and whether this worker's own file changed (then it must restart)."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,59}", branch) or ".." in branch:
        raise ValueError(f"not a branch name: {branch}")
    me = os.path.join(project, "scripts", "cloud_worker.py")
    with open(me, "rb") as f:
        before = f.read()
    git = lambda *a: subprocess.run(["git", "-C", project, *a], check=True, capture_output=True, text=True).stdout.strip()
    try:
        git("fetch", "-q", CODE_URL, f"+refs/heads/{branch}:refs/remotes/github/{branch}")
        if git("rev-parse", "--abbrev-ref", "HEAD") == branch:
            git("merge", "-q", "--ff-only", f"refs/remotes/github/{branch}")
        else:
            git("checkout", "-q", "-B", branch, f"refs/remotes/github/{branch}")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"couldn't get the code of {branch}: {e.stderr.strip()[-300:]}")
    with open(me, "rb") as f:
        return git("log", "-1", "--format=%h %s"), f.read() != before


def run(cloud, job_name):
    folder = os.path.join(WORK, job_name)
    os.makedirs(folder, exist_ok=True)
    zip_path = os.path.join(folder, job_name + ".zip")
    cloud.write_status(job_name, "running", "downloading the job")
    cloud.download(job_name + ".zip", zip_path)
    with zipfile.ZipFile(zip_path) as z:
        job = json.loads(z.read("job.json"))
        if job.get("kind") == "generate":
            z.extractall(folder)  # zipfile drops '..' and absolute paths
    if job.get("code"):
        commit, changed = use_code(job["code"])
        print(f"[{time.strftime('%H:%M')}] code: {job['code']} at {commit}", flush=True)
        if changed:  # this worker itself is new: start again with it, the job is still waiting
            cloud.write_status(job_name, "queued", f"the worker restarts with the new code ({commit})")
            os.execv(sys.executable, [sys.executable, "-u", os.path.abspath(__file__), *sys.argv[1:]])
        cloud.write_status(job_name, "running", f"code: {job['code']} at {commit}")
    gpu = freest_gpu()
    command = command_for(job, folder, zip_path, gpu)
    print(f"[{time.strftime('%H:%M')}] {job_name} on GPU {gpu}: {' '.join(command[2:])}", flush=True)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    process = subprocess.Popen(command, cwd=PROJECT_DIR, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    results, tail, line, last_report = [], [], b"", 0.0
    while True:
        char = process.stdout.read(1)
        if not char:
            break
        if char not in (b"\r", b"\n"):
            line += char
            continue
        text = line.decode(errors="replace").strip()
        line = b""
        if not text:
            continue
        tail = (tail + [text])[-30:]
        found = re.match(r"(?:Done|Result): (.+\.glb)", text)
        if found:
            results.append(os.path.join(PROJECT_DIR, found.group(1)) if not os.path.isabs(found.group(1)) else found.group(1))
        if char == b"\n":
            print(text, flush=True)
        if time.time() - last_report > 20:  # progress for Blender / cloud.py status
            cloud.write_status(job_name, "running", f"GPU {gpu}: {text[:150]}")
            last_report = time.time()
    if process.wait() != 0 or not results:
        cloud.write_status(job_name, "failed", " | ".join(tail[-4:])[:500])
        print(f"    failed: {tail[-3:]}", flush=True)
        return
    names = []
    for path in results:
        cloud.write_status(job_name, "running", f"uploading {os.path.basename(path)}")
        name = f"{job_name}__{os.path.basename(path)}"
        cloud.upload(path, name)
        names.append(name)
    cloud.write_status(job_name, "done", f"{len(names)} result(s)", names)
    print(f"    done: {names}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Run the jobs from the Sciebo cloud on this GPU server")
    parser.add_argument("--interval", type=int, default=30, help="seconds between looks into the cloud (default 30)")
    parser.add_argument("--once", action="store_true", help="look once, run what is there, then stop")
    args = parser.parse_args()
    cloud = Cloud()
    print(f"Watching the cloud folder {cloud.folder}/ every {args.interval} s (stop with Ctrl+C)", flush=True)
    while True:
        try:
            for name in sorted(cloud.list()):
                if name.endswith(".zip"):
                    job_name = name[:-4]
                    status = cloud.read_status(job_name)
                    if status is not None and status["state"] == "queued":
                        try:
                            run(cloud, job_name)
                        except Exception as e:
                            traceback.print_exc()
                            cloud.write_status(job_name, "failed", f"{type(e).__name__}: {e}"[:500])
        except Exception as e:  # the cloud or the network was away: try again next round
            print(f"[{time.strftime('%H:%M')}] cloud not reachable: {type(e).__name__}: {e}", flush=True)
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
