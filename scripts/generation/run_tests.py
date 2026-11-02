"""Run a series of generation tests on the uni GPU server by itself (e.g. overnight), one job at a time.

  1. organise   the local vision AI (organise_pictures.py, on the server) sorts each object's pictures:
                cut-outs, which side each shows, the front, the views to use together, what to leave out
  2. plan       from its answers, a list of tests per object: one picture vs. its views, with vs. without
                close-ups, the raw picture vs. its cut-out, resolution, steps, guidance, mode, mirror, seed
  3. run        one job at a time through the Sciebo cloud (so only one GPU of the shared server is used),
                in a round over the objects, most important tests first; no new job after --until
  4. record     every model goes to <out>/<object>/<test>.glb with a preview from 4 sides (Blender on the
                CPU), and one line in results.jsonl: pictures, the AI's answers, settings, time, faces.
                results.md is the table to look at; the "rating" column is for a person, and these ratings
                are what the settings can later be learned from.

It can be stopped and started again: tests already in results.jsonl are skipped.

Usage (TRELLIS.2 environment, on this PC; nothing runs on this PC's GPU):
    python scripts/generation/run_tests.py pictures/monster pictures/ninja "pictures/robot fly" \\
        --out outputs/tests/2026-11-03 --until 07:00
"""
import _paths  # noqa: F401  (the scripts of the other folders)

import argparse
import datetime
import glob
import io
import json
import os
import re
import shutil
import subprocess
import time
import traceback
import zipfile

from PIL import Image

from cloud import Cloud

PICTURE_TYPES = (".png", ".jpg", ".jpeg", ".webp")
RENDER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "render_preview.py")
BASE = {"resolution": "1024", "steps": 25}


def log(text):
    print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)


def safe(name):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")[:40]


def with_retries(work, what, tries=20, wait=60):
    """Runs work(); the cloud or the network may be away for a while at night."""
    for attempt in range(tries):
        try:
            return work()
        except Exception as e:
            log(f"{what}: {type(e).__name__}: {e} (try {attempt + 1}/{tries}, again in {wait} s)")
            time.sleep(wait)
    raise RuntimeError(f"{what} failed {tries} times")


def send(cloud, name, files, job):
    """Packs files ({name in zip: local path}) and job.json, uploads, returns the job's name."""
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as z:
        for packed, local in files.items():
            z.write(local, packed)
        z.writestr("job.json", json.dumps(job, indent=1))
    job_name = f"{name}_{time.strftime('%Y%m%d-%H%M%S')}"

    def upload():
        cloud.request("PUT", f"{cloud.folder}/{job_name}.zip", data=data.getvalue(), timeout=600)
        cloud.write_status(job_name, "queued", "waiting for the uni server")
    with_retries(upload, f"uploading {job_name}")
    return job_name


def wait(cloud, job_name, limit):
    """Waits until the job is done or failed (or `limit` seconds passed); returns its last status."""
    start, last, shown = time.time(), None, 0.0
    while time.time() - start < limit:
        try:
            status = cloud.read_status(job_name)
        except Exception as e:
            log(f"    can't read the status: {type(e).__name__}")
            status = None
        if status:
            line = f"{status['state']}: {status.get('message', '')}"[:110]
            if line != last and (not line.startswith("running: GPU") or time.time() - shown > 300):
                log(f"    {line}")
                shown = time.time()
            last = line
            if status["state"] in ("done", "failed"):
                return status
        time.sleep(30)
    return {"state": "timeout", "message": f"no answer after {limit // 60} min"}


def download(cloud, name, local):
    def get():
        cloud.download(name, local + ".part")
        os.replace(local + ".part", local)
    with_retries(get, f"downloading {name}")


# ---------------------------------------------------------------- 1. organise

def organise(cloud, folders, out):
    """The vision AI's answers for all objects (asks the server once; kept in <out>/organised/)."""
    target = os.path.join(out, "organised")
    if os.path.isfile(os.path.join(target, "organised.json")):
        return json.load(open(os.path.join(target, "organised.json")))
    files, names = {}, []
    for folder in folders:
        name = safe(os.path.basename(os.path.normpath(folder)))
        names.append(name)
        for path in sorted(glob.glob(os.path.join(folder, "*"))):
            if path.lower().endswith(PICTURE_TYPES):
                files[f"objects/{name}/{os.path.basename(path)}"] = path
    log(f"1. Organise: {len(files)} pictures of {', '.join(names)} -> the vision AI on the uni server")
    job_name = send(cloud, "organise", files, {"kind": "organise", "objects": names, "code": "main"})
    status = wait(cloud, job_name, 90 * 60)
    if status["state"] != "done" or not status.get("results"):
        log(f"   the vision AI didn't answer ({status['state']}: {status.get('message', '')[:200]})")
        return None
    archive = os.path.join(out, "organised.zip")
    download(cloud, status["results"][0], archive)
    with zipfile.ZipFile(archive) as z:
        z.extractall(target)
    result = json.load(open(os.path.join(target, "organised.json")))
    result["folders"] = {safe(os.path.basename(os.path.normpath(f))): f for f in folders}
    json.dump(result, open(os.path.join(target, "organised.json"), "w"), indent=1)
    log(f"   organised in {result.get('seconds', '?')} s")
    return result


def fallback(folders):
    """Without the vision AI: every picture as it is, the first one as the front, no views (marked)."""
    objects = []
    for folder in folders:
        files = sorted(f for f in glob.glob(os.path.join(folder, "*")) if f.lower().endswith(PICTURE_TYPES))
        objects.append({"object": safe(os.path.basename(os.path.normpath(folder))), "fallback": True,
                        "choice": {"front_file": files[0]}, "pictures": [{"file": os.path.basename(f)} for f in files]})
    return {"model": None, "objects": objects, "folders": {o["object"]: f for o, f in zip(objects, folders)}}


# ---------------------------------------------------------------- 2. plan

def plan(organised, out):
    """The tests: rounds over the objects, the most telling tests first."""
    crops = os.path.join(out, "organised")
    per_object = []
    for obj in organised["objects"]:
        name, choice = obj["object"], obj.get("choice", {})
        folder = organised["folders"][name]
        if obj.get("fallback"):
            front, views, closeups, raw = choice["front_file"], [], [], None
        else:
            if not choice.get("front_crop"):
                log(f"   {name}: the AI chose no front, skipped")
                continue
            front = os.path.join(crops, choice["front_crop"])
            views = [os.path.join(crops, c) for c in choice.get("view_crops", [])]
            closeups = [os.path.join(crops, c["crop"]) for c in obj["crops"] if not c.get("full_body")][:2]
            source = next((c["from"] for c in obj["crops"] if c["crop"] == choice["front_crop"]), None)
            raw = os.path.join(folder, source) if source else None
        multi = len(views) > 1
        colour = any(p.get("answer", {}).get("colours") in ("colour", "coloured") for p in obj.get("pictures", []))
        tests = [
            ("front_1024_s25", [front], {}),
            ("views_1024_s25", views, {"mode": "multidiffusion"}) if multi else None,
            ("raw_picture_1024_s25", [raw], {}) if raw else None,
            ("views_plus_closeups_1024_s25", views + closeups, {"mode": "multidiffusion"}) if multi and closeups else None,
            ("front_1536_s25", [front], {"resolution": "1536", "max_tokens": 40000}),
            ("views_stochastic_1024_s25", views, {"mode": "stochastic"}) if multi else None,
            ("front_1024_s50", [front], {"steps": 50}),
            ("front_1024_s12", [front], {"steps": 12}),
            ("front_guidance3", [front], {"guidance": 3.0}),
            ("front_guidance10", [front], {"guidance": 10.0}),
            ("views_mirror_1024_s25", views, {"mode": "multidiffusion", "mirror": True}) if multi and choice.get("symmetric") else None,
            ("front_texture_guidance3", [front], {"texture_guidance": 3.0}) if colour else None,
            ("front_seed7", [front], {"seed": 7}),
            ("views_1536_s25", views, {"mode": "multidiffusion", "resolution": "1536", "max_tokens": 40000}) if multi else None,
        ]
        per_object.append([{"id": f"{name}__{test}", "object": name, "test": test, "pictures": pictures,
                            "options": {**BASE, **options}} for test, pictures, options in (t for t in tests if t)])
    rounds = []
    for k in range(max((len(t) for t in per_object), default=0)):
        rounds += [tests[k] for tests in per_object if k < len(tests)]
    return rounds


# ---------------------------------------------------------------- 3. + 4. run and record

def preview(model, out):
    """4 views side by side, rendered by Blender on the CPU; None if Blender isn't there."""
    blender = shutil.which("blender")
    if not blender:
        return None
    base = out[:-4]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    subprocess.run([blender, "-b", "--python", RENDER, "--", model, base], env=env, capture_output=True, timeout=900)
    parts = [f"{base}_{k}.png" for k in range(4)]
    if not all(os.path.isfile(p) for p in parts):
        return None
    images = [Image.open(p) for p in parts]
    strip = Image.new("RGB", (sum(i.width for i in images), images[0].height))
    for k, image in enumerate(images):
        strip.paste(image, (k * image.width, 0))
    strip.save(out, quality=88)
    for p in parts:
        os.remove(p)
    return out


def faces(model):
    try:
        import trimesh
        return int(sum(len(g.faces) for g in trimesh.load(model, force="scene").geometry.values()))
    except Exception:
        return None


def run_test(cloud, test, organised, out):
    files = {f"picture{i}{os.path.splitext(p)[1].lower()}": p for i, p in enumerate(test["pictures"])}
    job = {"kind": "generate", "name": test["id"], "pictures": list(files), "options": test["options"]}
    started = time.time()
    job_name = send(cloud, safe(test["id"]), files, job)
    log(f"   sent as {job_name}")
    status = wait(cloud, job_name, 100 * 60)
    record = {"id": test["id"], "object": test["object"], "test": test["test"], "options": test["options"],
              "pictures": [os.path.relpath(p, out) if p.startswith(out) else p for p in test["pictures"]],
              "job": job_name, "state": status["state"], "message": status.get("message", ""),
              "minutes": round((time.time() - started) / 60, 1), "finished": time.strftime("%Y-%m-%d %H:%M"),
              "organised_by": organised.get("model")}
    obj = next((o for o in organised["objects"] if o["object"] == test["object"]), {})
    record["ai"] = {k: v for k, v in obj.get("choice", {}).items() if k != "raw"}
    if status["state"] == "done" and status.get("results"):
        folder = os.path.join(out, test["object"])
        os.makedirs(folder, exist_ok=True)
        model = os.path.join(folder, f"{test['test']}.glb")
        download(cloud, status["results"][0], model)
        record["model"] = os.path.relpath(model, out)
        record["faces"] = faces(model)
        shown = preview(model, os.path.join(folder, f"{test['test']}.jpg"))
        record["preview"] = os.path.relpath(shown, out) if shown else None
    return record


def table(records, out):
    """results.md: one row per test, with its preview and an empty rating column."""
    lines = ["# Generation tests", "", f"Updated {time.strftime('%Y-%m-%d %H:%M')}. Rate each result 1 (bad) to 5 "
             "(great) in the last column; these ratings are the training data for the settings.", "",
             "| Object | Test | Settings | Pictures | Minutes | Faces | Result | Preview | Rating |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in records:
        settings = ", ".join(f"{k}={v}" for k, v in r["options"].items())
        shown = f"![]({r['preview']})" if r.get("preview") else ""
        result = r.get("model") or f"{r['state']}: {r['message'][:80]}"
        lines.append(f"| {r['object']} | {r['test']} | {settings} | {len(r['pictures'])} | {r['minutes']} | "
                     f"{r.get('faces') or ''} | {result} | {shown} | |")
    open(os.path.join(out, "results.md"), "w").write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run generation tests on the uni server by themselves, one at a time")
    parser.add_argument("folders", nargs="+", help="one folder of pictures per object")
    parser.add_argument("--out", required=True)
    parser.add_argument("--until", default="07:00", help="start no new job after this time (HH:MM, default 07:00)")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    hour, minute = (int(x) for x in args.until.split(":"))
    now = datetime.datetime.now()
    until = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if until <= now:
        until += datetime.timedelta(days=1)
    log(f"Tests until {until:%Y-%m-%d %H:%M}, results in {args.out}")
    cloud = Cloud()

    organised = organise(cloud, args.folders, args.out) or fallback(args.folders)
    tests = plan(organised, args.out)
    json.dump(tests, open(os.path.join(args.out, "plan.json"), "w"), indent=1)
    log(f"2. Plan: {len(tests)} tests ({'organised by ' + organised['model'] if organised.get('model') else 'WITHOUT the vision AI'})")

    results = os.path.join(args.out, "results.jsonl")
    records = [json.loads(line) for line in open(results)] if os.path.isfile(results) else []
    done = {r["id"] for r in records}
    for k, test in enumerate(tests):
        if test["id"] in done:
            continue
        if datetime.datetime.now() >= until:
            log(f"It's {args.until}: no new job. {len(tests) - k} tests left for another night.")
            break
        log(f"3. Test {k + 1}/{len(tests)}: {test['id']}  {test['options']}  ({len(test['pictures'])} picture(s))")
        try:
            record = run_test(cloud, test, organised, args.out)
        except Exception as e:
            traceback.print_exc()
            record = {"id": test["id"], "object": test["object"], "test": test["test"], "options": test["options"],
                      "pictures": test["pictures"], "state": "error", "message": f"{type(e).__name__}: {e}"[:300],
                      "minutes": 0}
        records.append(record)
        with open(results, "a") as f:
            f.write(json.dumps(record) + "\n")
        table(records, args.out)
        log(f"   {record['state']} in {record['minutes']} min: {record.get('model') or record['message'][:120]}")
    log("Finished.")


if __name__ == "__main__":
    main()
