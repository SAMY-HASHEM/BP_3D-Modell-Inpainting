"""Run a series of generation tests on the uni GPU server by itself (e.g. overnight), one job at a time.

  1. organise   the local vision AI (organise_pictures.py, on the server) sorts each object's pictures:
                cut-outs, which side each shows, the front, the views to use together, what to leave out
  2. plan       from its answers, a list of tests per object: one picture vs. its views, with vs. without
                close-ups, the raw picture vs. its cut-out, resolution, steps, guidance, mode, mirror, seed
  3. run        one job at a time through the Sciebo cloud (so only one GPU of the shared server is used),
                in a round over the objects, most important tests first; no new job after --until
  4. refine    refine.json in --out may be written while it runs (after looking at the results): its tests,
                each with a "round" and a "why", go before the rest of the plan
  5. record     every model goes to <out>/<object>/<test>.glb with a preview from 4 sides (Blender on the
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
VIDEO_TYPES = (".mp4", ".mov", ".webm", ".avi", ".mkv", ".m4v")  # the vision AI looks at these itself
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
    """The vision AI's answers for all objects, kept in <out>/organised/. Objects it hasn't seen yet are
    sent to the server (once: a job already sent is waited for again after a restart, never sent twice)."""
    target = os.path.join(out, "organised")
    os.makedirs(os.path.join(target, "crops"), exist_ok=True)
    known = os.path.join(target, "organised.json")
    result = json.load(open(known)) if os.path.isfile(known) else {"model": None, "objects": [], "folders": {}}
    names = {safe(os.path.basename(os.path.normpath(f))): f for f in folders}
    result.setdefault("folders", {}).update(names)
    pending_file = os.path.join(out, "organise_jobs.json")
    pending = json.load(open(pending_file)) if os.path.isfile(pending_file) else {}
    waiting_for = {n for objects in pending.values() for n in objects}
    missing = [n for n in names if n not in {o["object"] for o in result["objects"]} and n not in waiting_for]
    if missing:
        files = {}
        for name in missing:
            for path in sorted(glob.glob(os.path.join(names[name], "*"))):
                if path.lower().endswith(PICTURE_TYPES + VIDEO_TYPES):
                    files[f"objects/{name}/{os.path.basename(path)}"] = path
        log(f"1. Organise: {len(files)} pictures/videos of {', '.join(missing)} -> the vision AI on the uni server")
        job_name = send(cloud, "organise", files, {"kind": "organise", "objects": missing, "code": "main"})
        pending[job_name] = missing
        json.dump(pending, open(pending_file, "w"), indent=1)
    for job_name, objects in list(pending.items()):
        log(f"1. Waiting for the vision AI: {job_name} ({', '.join(objects)})")
        status = wait(cloud, job_name, 120 * 60)
        del pending[job_name]
        json.dump(pending, open(pending_file, "w"), indent=1)
        if status["state"] != "done" or not status.get("results"):
            log(f"   the vision AI didn't answer ({status['state']}: {status.get('message', '')[:200]})")
            continue
        archive = os.path.join(out, f"{job_name}.zip")
        download(cloud, status["results"][0], archive)
        unpacked = os.path.join(out, job_name)
        with zipfile.ZipFile(archive) as z:
            z.extractall(unpacked)
        answer = json.load(open(os.path.join(unpacked, "organised.json")))
        for kind in ("crops", "frames"):
            os.makedirs(os.path.join(target, kind), exist_ok=True)
            for item in glob.glob(os.path.join(unpacked, kind, "*")):
                shutil.copy(item, os.path.join(target, kind))
        result["model"] = answer.get("model")
        result["objects"] = [o for o in result["objects"] if o["object"] not in objects] + answer["objects"]
        json.dump(result, open(known, "w"), indent=1)
        log(f"   organised {', '.join(objects)} in {answer.get('seconds', '?')} s")
    seen = {o["object"] for o in result["objects"]}
    for name in names:  # without the AI's answer: the pictures as they are (marked as fallback)
        if name not in seen:
            result["objects"].append(fallback([names[name]])["objects"][0])
    result["objects"] = [o for o in result["objects"] if o["object"] in names]
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

def plan(organised, out, folder="organised", tag=""):
    """The tests: rounds over the objects, the most telling tests first. `folder` holds the AI's cut-outs
    (another one after it chose again), `tag` marks its tests (e.g. "ai4")."""
    crops = os.path.join(out, folder)
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
            raw = None
            if source:  # the picture the front was cut from: in the folder, or a video frame the AI kept
                raw = next((p for p in (os.path.join(folder, source), os.path.join(crops, source)) if os.path.isfile(p)), None)
        every = [] if obj.get("fallback") else [os.path.join(crops, c) for c in obj.get("candidates", [])][:8]
        if every and views and views[0] in every:  # the front first
            every = [views[0]] + [v for v in every if v != views[0]]
        multi = len(views) > 1
        colour = any(p.get("answer", {}).get("colours") in ("colour", "coloured") for p in obj.get("pictures", []))
        tests = [
            ("front_1024_s25", [front], {}),
            ("views_1024_s25", views, {"mode": "multidiffusion"}) if multi else None,
            ("raw_picture_1024_s25", [raw], {}) if raw else None,
            ("views_plus_closeups_1024_s25", views + closeups, {"mode": "multidiffusion"}) if multi and closeups else None,
            ("all_views_stochastic_1024_s25", every, {"mode": "stochastic"}) if len(every) > len(views) else None,
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
        test_name = lambda test: f"{test}_{tag}" if tag else test
        per_object.append([{"id": f"{name}__{test_name(test)}", "object": name, "test": test_name(test), "pictures": pictures,
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
    job = {"kind": "generate", "name": test["id"], "pictures": list(files), "options": test["options"],
           "code": "main"}  # the worker brings itself to the newest code first (fixes made during the night)
    started = time.time()
    inflight_file = os.path.join(out, "inflight.json")  # the job on its way: after a restart, wait for it, never send twice
    inflight = json.load(open(inflight_file)) if os.path.isfile(inflight_file) else {}
    if test["id"] in inflight:
        job_name = inflight[test["id"]]
        log(f"   already sent as {job_name}: waiting for it")
    else:
        job_name = send(cloud, safe(test["id"]), files, job)
        json.dump({test["id"]: job_name}, open(inflight_file, "w"))
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
    if os.path.isfile(inflight_file):
        os.remove(inflight_file)
    return record


def table(records, out):
    """results.md: one row per test, with its preview and an empty rating column."""
    lines = ["# Generation tests", "", f"Updated {time.strftime('%Y-%m-%d %H:%M')}. Rate each result 1 (bad) to 5 "
             "(great) in the last column; these ratings are the training data for the settings.", "",
             "| Round | Object | Test | Settings | Pictures | Minutes | Faces | Result | Preview | Rating |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in records:
        settings = ", ".join(f"{k}={v}" for k, v in r["options"].items())
        shown = f"![]({r['preview']})" if r.get("preview") else ""
        result = r.get("model") or f"{r['state']}: {r['message'][:80]}"
        lines.append(f"| {r.get('round', 0)} | {r['object']} | {r['test']} | {settings} | {len(r['pictures'])} | {r['minutes']} | "
                     f"{r.get('faces') or ''} | {result} | {shown} | |")
    open(os.path.join(out, "results.md"), "w").write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run generation tests on the uni server by themselves, one at a time")
    parser.add_argument("folders", nargs="+", help="one folder of pictures per object")
    parser.add_argument("--out", required=True)
    parser.add_argument("--choices", default="organised",
                        help="folder in --out with the AI's choices to plan with (e.g. organised_v4 after --choose-again)")
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

    organised = organise(cloud, args.folders, args.out)
    if args.choices != "organised":  # the AI chose again (organise_pictures.py --choose-again): plan with that
        organised = {**json.load(open(os.path.join(args.out, args.choices, "organised.json"))), "folders": organised["folders"]}
        organised["objects"] = [o for o in organised["objects"] if o["object"] in organised["folders"]]
    tests = plan(organised, args.out, args.choices, args.choices.replace("organised_v", "ai") if args.choices != "organised" else "")
    json.dump(tests, open(os.path.join(args.out, "plan.json"), "w"), indent=1)
    log(f"2. Plan: {len(tests)} tests ({'organised by ' + organised['model'] if organised.get('model') else 'WITHOUT the vision AI'}; {sum(1 for o in organised['objects'] if o.get('fallback'))} object(s) without its answers)")

    results = os.path.join(args.out, "results.jsonl")
    records = [json.loads(line) for line in open(results)] if os.path.isfile(results) else []
    done = {r["id"] for r in records}
    # the same pictures with the same settings never run twice (e.g. a front the AI chose again unchanged)
    same = lambda t: (t["object"], tuple(os.path.basename(p) for p in t["pictures"]), json.dumps(t["options"], sort_keys=True))
    ran = {same(r) for r in records if r.get("state") == "done"}
    refine_file = os.path.join(args.out, "refine.json")
    while True:
        # refinement rounds: tests added while it runs (refine.json, written after looking at results) go first
        refine = []
        if os.path.isfile(refine_file):
            try:
                refine = json.load(open(refine_file))
            except json.JSONDecodeError:
                log("   refine.json can't be read right now; the plan goes on")
        queue = [t for t in refine + tests if t["id"] not in done and same(t) not in ran]
        if not queue:
            break
        if datetime.datetime.now() >= until:
            log(f"It's {args.until}: no new job. {len(queue)} tests left for another night.")
            break
        test = queue[0]
        kind = f"round {test['round']}" if test.get("round") else "plan"
        log(f"3. Test ({kind}, {len(queue)} waiting): {test['id']}  {test['options']}  ({len(test['pictures'])} picture(s))")
        try:
            record = run_test(cloud, test, organised, args.out)
        except Exception as e:
            traceback.print_exc()
            record = {"id": test["id"], "object": test["object"], "test": test["test"], "options": test["options"],
                      "pictures": test["pictures"], "state": "error", "message": f"{type(e).__name__}: {e}"[:300],
                      "minutes": 0}
        record["round"] = test.get("round", 0)
        record["why"] = test.get("why", "")
        records.append(record)
        done.add(test["id"])
        if record.get("state") == "done":
            ran.add(same(test))
        with open(results, "a") as f:
            f.write(json.dumps(record) + "\n")
        table(records, args.out)
        log(f"   {record['state']} in {record['minutes']} min: {record.get('model') or record['message'][:120]}")
    log("Finished.")


if __name__ == "__main__":
    main()
