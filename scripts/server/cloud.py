"""Send jobs to the uni GPU server through the Sciebo cloud, and get the results back.

The uni server can't be reached from home, but both the server and this PC can reach the Sciebo
cloud (Nextcloud, WebDAV). So the cloud folder inpainting-jobs/ is the mailbox: this PC puts a job
there, cloud_worker.py on the server runs it and puts the result next to it.

Only Python's standard library is used, so this runs with any python3 (and inside Blender).

Usage:
    python3 scripts/server/cloud.py setup                          # once, on the PC and once on the server
    python3 scripts/server/cloud.py send outputs/blender/tavern_job.zip          # a job from Blender
    python3 scripts/server/cloud.py generate pictures/torso/torso-front.png pictures/torso/torso-back.png \\
        --resolution 1024 --steps 25 --mirror --faces 4000000              # picture(s) -> model
    python3 scripts/server/cloud.py inpaint outputs/robot.glb pictures/robot.png --box -0.4 -0.5 -0.4 -0.1 0 0 \
        --seed 2 --steps 25                                 # regenerate an area, without Blender
    python3 scripts/server/cloud.py generate front.png --code voxhammer   # with the newest code of a branch (GitHub)
    python3 scripts/server/cloud.py status                         # all jobs and how far they are
    python3 scripts/server/cloud.py fetch                          # download finished results to outputs/cloud/

In the cloud: <job>.zip (the job), <job>.status (written by the worker: queued, running, done or
failed, with a message), <job>__<file>.glb (the results).
"""
import argparse
import base64
import getpass
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG = os.path.expanduser("~/.config/inpainting3d/cloud.json")
DEFAULT_URL = "https://uni-duisburg-essen.sciebo.de/remote.php/webdav"
FOLDER = "inpainting-jobs"
GENERATE_OPTIONS = {"resolution": str, "steps": int, "seed": int, "guidance": float, "texture_guidance": float,
                    "max_tokens": int, "faces": int, "mirror": bool, "mode": str}


# ---------------------------------------------------------------- talking to the cloud (WebDAV)

DIRECT = None  # an opener without proxy, once the proxy refused


def use_direct():
    global DIRECT
    DIRECT = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class Cloud:
    def __init__(self, config=None):
        if config is None:
            if not os.path.isfile(CONFIG):
                raise SystemExit(f"No cloud login yet: run  python3 scripts/server/cloud.py setup")
            with open(CONFIG) as f:
                config = json.load(f)
        self.url = config["url"].rstrip("/")
        self.folder = config.get("folder", FOLDER)  # e.g. a folder someone shared with you
        token = base64.b64encode(f"{config['user']}:{config['password']}".encode()).decode()
        self.auth = {"Authorization": f"Basic {token}"}

    def request(self, method, path, data=None, headers=None, timeout=60):
        url = f"{self.url}/{urllib.parse.quote(path)}"
        request = urllib.request.Request(url, data=data, method=method, headers={**self.auth, **(headers or {})})
        try:
            with (DIRECT.open if DIRECT else urllib.request.urlopen)(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.URLError as e:
            # a proxy that is down (as JupyterLab's on the uni server sometimes): go out directly instead
            if DIRECT or not isinstance(e.reason, ConnectionRefusedError) or not urllib.request.getproxies():
                raise
            print("    the proxy refuses: going to the cloud directly from now on", flush=True)
            use_direct()
            return self.request(method, path, data, headers, timeout)

    def list(self, folder=None):
        """Names of the files in a cloud folder."""
        folder = folder or self.folder
        body = self.request("PROPFIND", folder + "/", headers={"Depth": "1"})
        names = []
        for href in ET.fromstring(body).iter("{DAV:}href"):
            name = urllib.parse.unquote(href.text.rstrip("/").rsplit("/", 1)[-1])
            if name and name != folder.rstrip("/").rsplit("/", 1)[-1]:
                names.append(name)
        return names

    def make_folder(self, folder=None):
        folder = folder or self.folder
        try:
            self.request("MKCOL", folder + "/")
        except urllib.error.HTTPError as e:
            if e.code != 405:  # 405 = it exists already
                raise

    def upload(self, local, name, folder=None):
        folder = folder or self.folder
        with open(local, "rb") as f:
            self.request("PUT", f"{folder}/{name}", data=f.read(), timeout=600)

    def download(self, name, local, folder=None):
        folder = folder or self.folder
        data = self.request("GET", f"{folder}/{name}", timeout=600)
        with open(local, "wb") as f:
            f.write(data)

    def read_status(self, job, folder=None):
        folder = folder or self.folder
        try:
            return json.loads(self.request("GET", f"{folder}/{job}.status"))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            raise

    def write_status(self, job, state, message="", results=None, folder=None):
        folder = folder or self.folder
        status = {"state": state, "message": message, "results": results or [], "time": time.strftime("%H:%M:%S")}
        self.request("PUT", f"{folder}/{job}.status", data=json.dumps(status).encode())


# ---------------------------------------------------------------- the commands

def setup(args):
    """Asks for the cloud login (an app password, not the uni password), tests it, saves it privately."""
    print("Sciebo: avatar (top right) > Persönliche Einstellungen > Sicherheit > 'Neues App-Passwort erstellen'.")
    print("Use the login name and the password shown there.\n")
    url = input(f"WebDAV address [{DEFAULT_URL}]: ").strip() or DEFAULT_URL
    user = input("Login name (Anmeldename): ").strip()
    password = getpass.getpass("App password (not shown while typing): ").strip()
    folder = input(f"Cloud folder for the jobs [{FOLDER}] (someone shared theirs with you: its path): ").strip() or FOLDER
    cloud = Cloud({"url": url, "user": user, "password": password, "folder": folder})
    try:
        cloud.make_folder()
        cloud.list()
    except urllib.error.HTTPError as e:
        raise SystemExit(f"The cloud refused the login ({e.code}). Check the login name and the app password.")
    os.makedirs(os.path.dirname(CONFIG), exist_ok=True)
    with open(os.open(CONFIG, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as f:  # only you can read it
        json.dump({"url": url, "user": user, "password": password, "folder": folder}, f)
    print(f"Works. Saved in {CONFIG} (readable only by you); the cloud folder {folder}/ is ready.")


def send_zip(cloud, path):
    """Uploads a job .zip under a new, unique name; returns that name (without .zip)."""
    job = f"{os.path.splitext(os.path.basename(path))[0]}_{time.strftime('%Y%m%d-%H%M%S')}"
    cloud.upload(path, job + ".zip")
    cloud.write_status(job, "queued", "waiting for the uni server")
    return job


def send(args):
    cloud = Cloud()
    job = send_zip(cloud, args.job)
    print(f"Sent as {job}. Follow it with:  python3 scripts/server/cloud.py status")


def generate(args):
    """Packs the picture(s) and the settings into a job and sends it."""
    options = {k: getattr(args, k) for k in GENERATE_OPTIONS if getattr(args, k) not in (None, False)}
    name = os.path.splitext(os.path.basename(args.pictures[0]))[0]
    os.makedirs(os.path.join(PROJECT_DIR, "outputs", "cloud"), exist_ok=True)
    path = os.path.join(PROJECT_DIR, "outputs", "cloud", f"{name}_generate.zip")
    pictures = [f"picture{i}{os.path.splitext(p)[1].lower()}" for i, p in enumerate(args.pictures)]
    with zipfile.ZipFile(path, "w") as z:
        for local, packed in zip(args.pictures, pictures):
            z.write(local, packed)
        z.writestr("job.json", json.dumps({"kind": "generate", "name": name, "pictures": pictures, "options": options,
                                           **({"code": args.code} if args.code else {})}, indent=2))
    cloud = Cloud()
    job = send_zip(cloud, path)
    print(f"Sent as {job} ({len(pictures)} picture(s), {options}). Follow it with:  python3 scripts/server/cloud.py status")


def inpaint(args):
    """Packs a model, an area (a box in the model's coordinates, or a Blender area .npy) and a picture into
    the same job .zip that Blender's add-on makes, and sends it."""
    import numpy as np  # only this command needs them (TRELLIS.2 environment)
    import trimesh
    if args.area:
        points = np.load(args.area)
    elif args.box:  # filled with points, about 2 per voxel of TRELLIS.2's 64 grid
        size = np.ptp(trimesh.load(args.model).bounds, 0).max()
        low, high = np.minimum(args.box[:3], args.box[3:]), np.maximum(args.box[:3], args.box[3:])
        axes = [np.linspace(low[i], high[i], max(2, int((high[i] - low[i]) / size * 128))) for i in range(3)]
        points = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    else:
        raise SystemExit("give the area: --box X0 Y0 Z0 X1 Y1 Z1 (model coordinates) or --area file.npy")
    name = re.sub("_+", "_", os.path.splitext(os.path.basename(args.model))[0][:40])  # "__" marks results
    folder = os.path.join(PROJECT_DIR, "outputs", "cloud")
    os.makedirs(folder, exist_ok=True)
    area_file, path = os.path.join(folder, f"{name}_area.npy"), os.path.join(folder, f"{name}_job.zip")
    np.save(area_file, points)
    picture = "picture" + os.path.splitext(args.picture)[1].lower()
    job = {"model": "model.glb", "points": "area.npy", "picture": picture, "object": name,
           "seed": args.seed, "grow": args.grow, "steps": args.steps, **({"code": args.code} if args.code else {})}
    if args.variant:  # e.g. "inversion,start=random,soft=2"
        job["variants"] = []
        for text in args.variant:
            method, *settings = text.split(",")
            version = {"method": method}
            for setting in settings:
                key, _, value = setting.partition("=")
                version[key] = int(value) if value.isdigit() else float(value) if value.replace(".", "", 1).isdigit() else value
            job["variants"].append(version)
    if args.seeds:
        job["seeds"] = args.seeds
    source = "source" + os.path.splitext(args.source)[1].lower() if args.source else None
    if source:
        job["source"] = source
    if args.picture_mask:
        job["picture_mask"] = "picture_mask.png"
    with zipfile.ZipFile(path, "w") as z:
        z.write(args.model, "model.glb")
        z.write(area_file, "area.npy")
        z.write(args.picture, picture)
        if source:
            z.write(args.source, source)
        if args.picture_mask:
            z.write(args.picture_mask, "picture_mask.png")
        z.writestr("job.json", json.dumps(job, indent=2))
    cloud = Cloud()
    sent = send_zip(cloud, path)
    print(f"Sent as {sent} ({job.get('variants', 'repaint')}, seeds {job.get('seeds', [args.seed])}, grow {args.grow}, "
          f"{args.steps} steps). Follow it with:  python3 scripts/server/cloud.py status")


def status(args):
    cloud = Cloud()
    names = cloud.list()
    jobs = sorted(n[:-4] for n in names if n.endswith(".zip"))
    if not jobs:
        print("No jobs in the cloud.")
    for job in jobs:
        s = cloud.read_status(job) or {"state": "?", "message": ""}
        results = [n for n in names if n.startswith(job + "__")]
        print(f"{s['state']:>8}  {job}  {s.get('time', '')}  {s['message']}" + (f"  [{len(results)} result(s)]" if results else ""))


def fetch(args):
    cloud = Cloud()
    out = os.path.join(PROJECT_DIR, "outputs", "cloud")
    os.makedirs(out, exist_ok=True)
    for name in cloud.list():
        if "__" in name and name.endswith(".glb") and (args.job is None or name.startswith(args.job + "__")):
            local = os.path.join(out, name)
            if not os.path.exists(local):
                print(f"downloading {name}...")
                cloud.download(name, local)
                print(f"    -> {os.path.relpath(local, PROJECT_DIR)}")


def main():
    parser = argparse.ArgumentParser(description="Uni GPU server jobs through the Sciebo cloud")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("setup", help="save the cloud login (once per computer)").set_defaults(run=setup)
    p = sub.add_parser("send", help="send a job .zip (e.g. from Blender's 'Save job for server')")
    p.add_argument("job")
    p.set_defaults(run=send)
    p = sub.add_parser("generate", help="send picture(s) to make a model (one picture: generate.py, more: multi_image.py)")
    p.add_argument("pictures", nargs="+")
    p.add_argument("--resolution", choices=["512", "1024", "1536"])
    p.add_argument("--steps", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--guidance", type=float)
    p.add_argument("--texture-guidance", dest="texture_guidance", type=float)
    p.add_argument("--max-tokens", dest="max_tokens", type=int)
    p.add_argument("--faces", type=int)
    p.add_argument("--mirror", action="store_true")
    p.add_argument("--mode", choices=["multidiffusion", "stochastic"])
    p.add_argument("--code", metavar="BRANCH", help="run it with the newest code of this branch (from GitHub)")
    p.set_defaults(run=generate)
    p = sub.add_parser("inpaint", help="send an inpainting job without Blender (TRELLIS.2 environment)")
    p.add_argument("model")
    p.add_argument("picture", help="picture that guides the new area")
    p.add_argument("--box", type=float, nargs=6, metavar=("X0", "Y0", "Z0", "X1", "Y1", "Z1"),
                   help="the area as a box in the model's coordinates (glTF: Y up)")
    p.add_argument("--area", help="or the area .npy of a Blender job")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--grow", type=int, default=2)
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--seeds", type=int, nargs="+", help="several versions with these seeds")
    p.add_argument("--variant", action="append", metavar="METHOD[,KEY=VALUE...]",
                   help="a version to make, e.g. repaint or inversion,start=random,soft=2 (repeat to compare)")
    p.add_argument("--source", help="inversion: a picture of the original model (default: the picture)")
    p.add_argument("--picture-mask", dest="picture_mask",
                   help="the 2D mask of the edit (white = changed); --source is then the original's render of that view")
    p.set_defaults(run=inpaint)
    p.add_argument("--code", metavar="BRANCH", help="run it with the newest code of this branch (from GitHub)")
    sub.add_parser("status", help="list the jobs and how far they are").set_defaults(run=status)
    p = sub.add_parser("fetch", help="download finished results to outputs/cloud/")
    p.add_argument("job", nargs="?", help="only this job (default: all)")
    p.set_defaults(run=fetch)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
